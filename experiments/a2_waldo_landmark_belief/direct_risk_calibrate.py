"""CPU-only T1 score-gate fitting; ONE unique disjoint verification decision."""
from collections import Counter
import json,time
from pathlib import Path
import numpy as np
import torch
from .paths import ROOT,DATA,save,sha,bootstrap
from .collected_snapshot import BASE
from .direct_risk_head import DirectRiskHead
from .direct_risk_protocol import operating_points,select_point
from .direct_risk_confirmation import CONFIRMATION_CONFIG,confirmation_code_hashes,validate_completed_episode
from .revision_confirmation import immutable_json,confirmation_entries
from .revision_calibrate import save_packet,load_packet
from .revision_fusion import proposal_distribution,bounded_belief_mix
from .risk_probe import public_risk_features
from .direct_risk_numeric import audit_gpu_proposal
from .calibration_protocol import calibration_fold,group_summary,verify_gate


@torch.no_grad()
def make_packet(row,head):
    spatial=row['public']['spatial'];prediction=row['prediction'];risk=row['risk'];base=spatial['base_logits'].float();q=risk['q'].float()
    available=bool(risk['available']);numeric=audit_gpu_proposal(spatial,prediction,q,available)
    def close(a,b):return a.shape==b.shape and bool(torch.allclose(a.float(),b.float(),atol=1e-5,rtol=1e-5))
    features=public_risk_features(base[None],q[None],spatial['grid_world'][None],prediction['benefit_logit'].reshape(1).sigmoid(),torch.tensor([available]))
    if not close(features,risk['features']) or not close(head(risk['features']),risk['head_logits']):raise ValueError('Risk features/head do not reproduce')
    corrected=head.corrected_logits(risk['head_logits'])
    scores=risk['scores'].float()
    if not close(corrected,risk['corrected_logits']) or not close(corrected[0].softmax(-1),scores):raise ValueError('Actual GPU T1scores differ from frozen head')
    fixed=risk['fixed_action_logits'].float();expected=bounded_belief_mix(base,q,.25) if available else base
    if not close(fixed,expected):raise ValueError('Stored GPU fixedaction differs from defined update')
    if scores.shape!=(3,) or not torch.isfinite(scores).all() or (scores<0).any() or not torch.isclose(scores.sum(),torch.tensor(1.),atol=1e-6):raise ValueError('Invalid actual class scores')
    target=row['labels']['target_xy'].float();grid=spatial['grid_world'].float();distance=(grid-target).norm(dim=-1)
    return dict(group=row['target_instance'],fold=row['fold'],episode=row['episode'],step=row['step'],scores=scores,
        available=bool(available),changed=bool(available and not torch.equal(base,fixed)),before_ne=float(distance[base.argmax()]),action_ne=float(distance[fixed.argmax()]),numeric_proposal_audit=numeric)


def evaluate_risk(packets,parameters):
    rows=[]
    for p in packets:
        gain=float(p['scores'][1]);harm=float(p['scores'][2])
        update=p['available'] and p['changed'] and gain>=parameters['minimum_gain'] and harm<=parameters['maximum_harm']
        before=p['before_ne'];after=p['action_ne'] if update else before
        rows.append(dict(group=p['group'],a0_correct=before<=20,a2_correct=after<=20,a0_ne=before,a2_ne=after,changed=update))
    result=group_summary(rows)
    result['metric']='Grid argmax R@1 within20m on real frozen-A0 closed-loop observations; actual GPU fixed.25action and T1scores; target equal; NOT navigation SR'
    return result


def fit_choice(packets):
    if any(p['fold']!='fit' or calibration_fold(*p['group'])!='fit' for p in packets):raise ValueError('FIT selection cannot consume VERIFY targets')
    if not packets:return [],None
    scores=np.stack([p['scores'].numpy() for p in packets]);available=np.array([p['available'] for p in packets],bool)
    ids=[dict(map_name=p['group'][0],object_id=p['group'][1]) for p in packets]
    points=operating_points(scores,available,ids,np.array([p['before_ne'] for p in packets]),np.array([p['action_ne'] for p in packets]))
    for point in points:point['summary']=evaluate_risk(packets,point['parameters'])
    return points,select_point(points,dev=False)


def load_packets(root):
    root=Path(root);directory=root/'confirmation';output=root/'calibration/packets'
    manifest=json.loads((directory/'manifest.json').read_text());signature=json.loads((directory/'signature.json').read_text())
    population=json.loads((root/'confirmation_population.json').read_text());source=json.loads((BASE/'source_manifest.json').read_text())
    if (manifest['status']!='direct_risk_independent_A0_confirmation_complete' or manifest['target_groups']!=256
        or manifest['signature_sha256']!=sha(directory/'signature.json') or signature['code_sha256']!=confirmation_code_hashes()
        or signature['population_sha256']!=sha(root/'confirmation_population.json') or signature['protocol_sha256']!=sha(root/'confirmation_protocol.json')
        or signature['sensor_signature_sha256']!=sha(root/'sensor_signature.json')
        or signature['protocol_preobservation_revision_sha256']!=sha(root/'protocol_preobservation_revision.json')
        or signature['checkpoint_sha256']!=sha(root/'head_training/best.pt') or signature['rank_checkpoint_sha256']!=sha(BASE/'revision_v1/training/best.pt')
        or signature['training_verification_sha256']!=sha(root/'head_training/artifact_verification.json')
        or signature['preflight_sha256']!=sha(root/'runtime_preflight.json')
        or population['reserve_sha256']!=sha(BASE/'revision_v2_confirmation_reserve.json') or population['source_manifest_sha256']!=sha(BASE/'source_manifest.json')):
        raise ValueError('Complete independent risk confirmation lineage changed')
    expected_entries=confirmation_entries(json.loads((BASE/'revision_v2_confirmation_reserve.json').read_text()),source)
    if population['entries']!=expected_entries:raise ValueError('Fixed new query population changed')
    from .prepare import cache_signature
    if json.loads((root/'sensor_signature.json').read_text())!=cache_signature():raise ValueError('Live sensor/annotation/foundation dependencies changed')
    expected={e['index']:e for e in expected_entries}
    bootstrap()
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.closedloop import PublicCatalog,ClosedLoopProtocol
    dataset=CityNavDataset(DATA,'train_seen');catalog=CityReferCatalog(DATA/'cityrefer')
    if sha(dataset.annotation_path)!=source['annotation_sha256']:raise ValueError('Confirmation source annotation changed')
    builder=PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28)
    seen=set();episode_rows=[]
    for item in manifest['episodes']:
        path=directory/item['complete_file']
        if Path(item['complete_file']).name!=item['complete_file'] or sha(path)!=item['sha256'] or sha(path.with_suffix('.sha.json'))!=item['sidecar_sha256']:
            raise ValueError('Confirmation episode/commit changed')
        if item['index'] in seen or item['index'] not in expected:raise ValueError('Duplicate/unknown confirmation episode')
        seen.add(item['index']);entry=expected[item['index']];record=dataset[item['index']];transform=builder.transform(record.map_name)
        if (record.map_name,record.object_id,record.description_id)!=(entry['map_name'],entry['object_id'],entry['description_id']):raise ValueError('Live confirmation source identity changed')
        info=validate_completed_episode(path,directory,entry,manifest['signature_sha256'],record,
            (transform.x_min,transform.y_min,transform.x_max,transform.y_max),GROUND_LEVEL[record.map_name],ClosedLoopProtocol())
        episode_rows.extend(info['rows'])
    def identities(rows):return Counter((e['filename'],e['sha256'],e['episode'],e['step'],e['fold']) for e in rows)
    if seen!=set(expected) or identities(episode_rows)!=identities(manifest['entries']) or len({e['filename'] for e in manifest['entries']})!=len(manifest['entries']):
        raise ValueError('Confirmation raw population incomplete/duplicate')
    output.mkdir(parents=True,exist_ok=True)
    packet_signature=dict(source_manifest_sha256=sha(directory/'manifest.json'),checkpoint_sha256=signature['checkpoint_sha256'],
        code_sha256=sha(Path(__file__)),temperature=1.,scores='actualGPU runtimeT1; no temperature estimator',action='actualGPU fixed.25 logits')
    immutable_json(packet_signature,output/'signature.json');digest=sha(output/'signature.json')
    state=torch.load(root/'head_training/best.pt',map_location='cpu',weights_only=False);m=state['model']
    head=DirectRiskHead(m['mean'],m['std'],m['class_weights']).eval().requires_grad_(False);head.load_state_dict(m)
    packets=[];packet_entries=[]
    for item in manifest['entries']:
        raw=directory/item['filename'];saved=output/item['filename'];entry=expected[item['episode']]
        if Path(item['filename']).name!=item['filename'] or sha(raw)!=item['sha256']:raise ValueError('Raw source changed')
        if saved.exists() and saved.with_suffix('.sha.json').exists():p=load_packet(saved,digest,item['sha256'])
        else:
            row=torch.load(raw,map_location='cpu',weights_only=False);p=make_packet(row,head)
            p.update(source_row_sha256=item['sha256'],signature_sha256=digest);save_packet(p,saved)
        if (p['group']!=[entry['map_name'],entry['object_id']] or p['fold']!=entry['fold'] or p['episode']!=entry['index'] or p['step']!=item['step']
            or p['fold']!=calibration_fold(*p['group'])):raise ValueError('Derived risk packet identity/fold changed')
        packets.append(p);packet_entries.append(dict(filename=item['filename'],sha256=sha(saved),sidecar_sha256=sha(saved.with_suffix('.sha.json'))))
    if len({tuple(p['group']) for p in packets})!=256:raise ValueError('Missing independent target groups')
    save(dict(signature_sha256=digest,entries=packet_entries),output/'manifest.json')
    return packets,signature


def main():
    torch.set_num_threads(2);root=BASE/'revision_v2';output=root/'calibration';started=time.perf_counter()
    if (output/'candidate_calibration.json').exists():raise ValueError('Unique new decision already exists; inspect instead of refitting')
    from .direct_risk_train import verify_completed_training
    verification=verify_completed_training()
    if not verification['checkpoint_eligible']:raise ValueError('Ineligible head cannot use confirmation')
    config=json.loads((root/'confirmation_protocol.json').read_text())
    if config!=CONFIRMATION_CONFIG or config['temperatures']!=[1.]:raise ValueError('Final pre-observation T1protocol changed')
    output.mkdir(parents=True,exist_ok=True)
    implementations=('direct_risk_calibrate.py','direct_risk_confirmation.py','direct_risk_runtime.py','direct_risk_head.py',
                     'direct_risk_protocol.py','direct_risk_numeric.py','revision_calibrate.py','calibration_protocol.py','risk_probe.py','revision_fusion.py')
    analysis=dict(checkpoint_sha256=sha(root/'head_training/best.pt'),rank_checkpoint_sha256=sha(BASE/'revision_v1/training/best.pt'),
        protocol_sha256=sha(root/'confirmation_protocol.json'),temperature=1.,temperature_fitting=False,
        implementation={n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in implementations})
    immutable_json(analysis,output/'analysis_signature.json');packets,source_signature=load_packets(root)
    fit=[p for p in packets if p['fold']=='fit'];verify=[p for p in packets if p['fold']=='verify']
    if {tuple(p['group']) for p in fit}&{tuple(p['group']) for p in verify}:raise ValueError('Risk fit/verify targets overlap')
    choices,chosen=fit_choice(fit)
    selected=chosen['parameters'] if chosen else None
    decision=dict(temperature=1.,temperature_fitting=False,choices=choices,selected_index=chosen['index'] if chosen else None,
        selected_parameters=selected,selection='FITonly fixed30quantiles and declared tie-break; ONE subsequent VERIFY; no runnerup')
    immutable_json(decision,output/'fit_decision.json')
    # No heldout aggregate is inspected before the unique FITdecision is sealed.
    stats=evaluate_risk(verify,selected) if selected is not None else None
    risk=verify_gate(stats) if stats is not None else dict(passed=False,checks=dict(fit_candidate_exists=False))
    enabled=bool(chosen is not None and risk['passed'])
    result=dict(status='direct_risk_candidate_requires_real_closed_loop_pilot',gate_final=False,enabled=enabled,
        minimum_gain=selected['minimum_gain'] if enabled else 1.,maximum_harm=selected['maximum_harm'] if enabled else 0.,
        temperature=1.,temperature_fitting=False,fixed_action=.25,checkpoint_sha256=analysis['checkpoint_sha256'],rank_checkpoint_sha256=analysis['rank_checkpoint_sha256'],
        fit_selected_parameters=selected,verification=stats,verify_gate=risk,fit_groups=len({tuple(p['group']) for p in fit}),verify_groups=len({tuple(p['group']) for p in verify}),
        no_verify_reselection=True,benchmark_used=False,observation_source=source_signature['observation_source'],
        analysis_signature_sha256=sha(output/'analysis_signature.json'),fit_decision_sha256=sha(output/'fit_decision.json'),
        confirmation_manifest_sha256=sha(root/'confirmation/manifest.json'),packet_manifest_sha256=sha(output/'packets/manifest.json'),
        training_verification_sha256=sha(root/'head_training/artifact_verification.json'),seconds=time.perf_counter()-started)
    save(result,output/'candidate_calibration.json');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
