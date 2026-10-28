"""One fit-only gate choice, one disjoint verification; no benchmark access."""
import argparse
from collections import Counter,defaultdict
import json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .paths import ROOT,save,sha
from .collected_snapshot import BASE
from .calibration_protocol import calibration_fold,group_summary,verify_gate
from .revision_confirmation import CONFIRMATION_CONFIG,immutable_json,confirmation_code_hashes
from .revision_fusion import proposal_distribution,bounded_belief_mix,MAX_GATE


def revision_packet(row,*,device='cpu'):
    s={k:v.to(device) for k,v in row['public']['spatial'].items()};out={k:v.to(device) for k,v in row['prediction'].items()}
    q,available=proposal_distribution(s,out['scores'],out['candidate_match_logits'],out['reliable'])
    return dict(group=row['target_instance'],fold=row['fold'],episode=row['episode'],step=row['step'],
                q=q.detach().cpu(),available=available,base=s['base_logits'].detach().cpu(),grid=s['grid_world'].detach().cpu(),
                target=row['labels']['target_xy'].detach().cpu(),benefit_logit=out['benefit_logit'].detach().cpu())


def fit_benefit_temperature(packets):
    groups=defaultdict(list);positive_groups=set();negative_groups=set();positives=negatives=0
    for p in packets:
        if p['fold']!='fit' or calibration_fold(*p['group'])!='fit':raise ValueError('Temperature fitting cannot inspect verification targets')
        if not p['available']:continue
        distance=(p['grid']-p['target']).norm(dim=-1)
        helpful=bool((p['q']*distance).sum()+1.<(p['base'].softmax(-1)*distance).sum())
        key=tuple(p['group']);groups[key].append((p['benefit_logit'].float(),float(helpful)))
        positives+=int(helpful);negatives+=int(not helpful)
        (positive_groups if helpful else negative_groups).add(key)
    c=CONFIRMATION_CONFIG
    sufficient=(positives>=c['min_temperature_positive'] and negatives>=c['min_temperature_negative']
                and len(positive_groups)>=c['min_temperature_positive_groups'] and len(negative_groups)>=c['min_temperature_negative_groups'])
    results=[]
    for t in c['temperatures']:
        group_nll=[]
        for values in groups.values():
            logits=torch.stack([v[0] for v in values]);truth=torch.tensor([v[1] for v in values])
            group_nll.append(float(F.binary_cross_entropy_with_logits(logits/t,truth)))
        results.append(dict(temperature=t,nll=sum(group_nll)/len(group_nll) if group_nll else None))
    temperature=min(results,key=lambda r:r['nll'])['temperature'] if sufficient else 1.
    return temperature,dict(sufficient=sufficient,positive_labels=positives,negative_labels=negatives,
                            positive_groups=len(positive_groups),negative_groups=len(negative_groups),choices=results,
                            target='Proposal expected distance improves A0 by>1m; fit available evidence only; equal target-group NLL')


@torch.no_grad()
def evaluate_gate(packets,choice,temperature,*,device='cuda'):
    rows=[]
    for p in packets:
        base=p['base'].to(device);q=p['q'].to(device);grid=p['grid'].to(device);target=p['target'].to(device)
        probability=float(torch.sigmoid(p['benefit_logit'].to(device)/temperature))
        gate=MAX_GATE*probability*choice['scale'] if p['available'] and probability>=choice['threshold'] else 0.
        mixed=bounded_belief_mix(base,q,gate)
        before=float((grid[base.argmax()]-target).norm());after=float((grid[mixed.argmax()]-target).norm())
        rows.append(dict(group=p['group'],a0_correct=before<=20,a2_correct=after<=20,a0_ne=before,a2_ne=after,changed=not torch.equal(base,mixed)))
    result=group_summary(rows)
    result['metric']='Grid argmax R@1 within20m on real frozen-A0 closed-loop confirmation observations; equal target weights; not navigation SR'
    return result


def save_packet(packet,path):
    from .train import atomic_checkpoint
    path=Path(path);atomic_checkpoint(packet,path)
    save(dict(packet_sha256=sha(path),source_row_sha256=packet['source_row_sha256'],signature_sha256=packet['signature_sha256']),path.with_suffix('.sha.json'))


def load_packet(path,signature_digest,source_digest):
    path=Path(path);bound=json.loads(path.with_suffix('.sha.json').read_text())
    if bound['packet_sha256']!=sha(path):raise ValueError('Calibration packet checksum mismatch')
    if bound['signature_sha256']!=signature_digest or bound['source_row_sha256']!=source_digest:raise ValueError('Packet sidecar provenance changed')
    packet=torch.load(path,map_location='cpu',weights_only=False)
    if packet['signature_sha256']!=signature_digest or packet['source_row_sha256']!=source_digest:raise ValueError('Packet contents provenance changed')
    return packet


def load_packets(directory,output):
    manifest=json.loads((directory/'manifest.json').read_text());signature=json.loads((directory/'signature.json').read_text())
    if manifest['status']!='independent_A0_confirmation_complete' or manifest['target_groups']!=256 or manifest['signature_sha256']!=sha(directory/'signature.json'):
        raise ValueError('Full independent confirmation manifest required')
    if signature['code_sha256']!=confirmation_code_hashes():raise ValueError('Confirmation collection/runtime code changed')
    population=json.loads((BASE/'revision_v1/confirmation_population.json').read_text())
    if signature['population_sha256']!=sha(BASE/'revision_v1/confirmation_population.json'):raise ValueError('Reserved query population changed')
    expected={e['index']:e for e in population['entries']};episode_rows=[];seen=set()
    for item in manifest['episodes']:
        path=directory/item['complete_file']
        if sha(path)!=item['sha256']:raise ValueError('Confirmation episode file changed')
        info=json.loads(path.read_text());entry=info['entry']
        if entry!=expected[entry['index']] or entry['index'] in seen or info['signature_sha256']!=manifest['signature_sha256']:
            raise ValueError('Confirmation episode identity/provenance changed')
        seen.add(entry['index']);episode_rows.extend(info['rows'])
    def identities(rows):return Counter((e['filename'],e['sha256'],e['episode'],e['step'],e['fold']) for e in rows)
    if seen!=set(expected) or identities(episode_rows)!=identities(manifest['entries']):
        raise ValueError('Confirmation does not cover all reserved source rows')
    if len({e['filename'] for e in manifest['entries']})!=len(manifest['entries']):raise ValueError('Duplicate confirmation rows')
    output.mkdir(parents=True,exist_ok=True)
    prediction_signature=dict(source_manifest_sha256=sha(directory/'manifest.json'),checkpoint_sha256=signature['checkpoint_sha256'],
                              calibration_code_sha256=sha(Path(__file__)),fusion_code_sha256=sha(ROOT/'experiments/a2_waldo_landmark_belief/revision_fusion.py'))
    immutable_json(prediction_signature,output/'signature.json');digest=sha(output/'signature.json');packets=[]
    for item in manifest['entries']:
        path=directory/item['filename']
        if sha(path)!=item['sha256']:raise ValueError('Confirmation source row changed')
        saved=output/item['filename']
        if saved.exists() and saved.with_suffix('.sha.json').exists():
            packet=load_packet(saved,digest,item['sha256'])
        else:
            row=torch.load(path,map_location='cpu',weights_only=False);entry=expected[item['episode']]
            if row['signature_sha256']!=manifest['signature_sha256'] or row['episode']!=entry['index'] or row['step']!=item['step'] or row['target_instance']!=[entry['map_name'],entry['object_id']] or row['fold']!=entry['fold']:
                raise ValueError('Confirmation private identity/observation mismatched')
            packet=revision_packet(row,device='cuda');packet.update(source_row_sha256=item['sha256'],signature_sha256=digest)
            save_packet(packet,saved)
        entry=expected[item['episode']]
        if packet['group']!=[entry['map_name'],entry['object_id']] or packet['fold']!=entry['fold'] or packet['episode']!=entry['index'] or packet['step']!=item['step']:
            raise ValueError('Cached calibration packet identity changed')
        packets.append(packet)
    if len({tuple(p['group']) for p in packets})!=256:raise ValueError('Missing independent target observations')
    save(dict(signature_sha256=digest,entries=[dict(filename=e['filename'],sha256=sha(output/e['filename']),sidecar_sha256=sha((output/e['filename']).with_suffix('.sha.json'))) for e in manifest['entries']]),output/'manifest.json')
    return packets,signature


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--confirmation',type=Path,default=BASE/'revision_v1/confirmation');parser.add_argument('--output',type=Path,default=BASE/'revision_v1/calibration');a=parser.parse_args()
    from .revision_artifacts import verify_revision_training
    state,verification=verify_revision_training();checkpoint=BASE/'revision_v1/training/best.pt'
    if not state['checkpoint_eligible']:raise ValueError('Internal candidate lacks positive gain; do not use confirmation')
    torch.set_num_threads(2);started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'candidate_calibration.json').exists():raise ValueError('Unique calibrated candidate already exists; inspect instead of refitting')
    config=json.loads((BASE/'revision_v1/confirmation_protocol.json').read_text())
    if config!=CONFIRMATION_CONFIG:raise ValueError('Predeclared confirmation protocol changed')
    signature=dict(checkpoint_sha256=sha(checkpoint),protocol_sha256=sha(BASE/'revision_v1/confirmation_protocol.json'),
                   implementation={n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in ('revision_calibrate.py','revision_runtime.py','revision_fusion.py','calibration_protocol.py')})
    immutable_json(signature,a.output/'analysis_signature.json')
    packets,source_signature=load_packets(a.confirmation,a.output/'packets')
    if source_signature['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Confirmation used different checkpoint')
    fit=[p for p in packets if p['fold']=='fit'];verify=[p for p in packets if p['fold']=='verify']
    if {tuple(p['group']) for p in fit}&{tuple(p['group']) for p in verify}:raise ValueError('Fit/verify targets overlap')
    temperature,temp_report=fit_benefit_temperature(fit);choices=[];viable=[]
    for scale in config['scales']:
        for threshold in config['thresholds']:
            choice=dict(scale=scale,threshold=threshold);summary=evaluate_gate(fit,choice,temperature)
            choices.append(dict(parameters=choice,summary=summary));i=len(choices)-1
            if (summary['mean_r1_difference']>0 and summary['mean_ne_difference_m']<=0 and summary['supported_groups']>=config['min_fit_supported_groups']
                and summary['broken_group_rate'] is not None and summary['broken_group_rate']<=config['max_fit_broken_group_rate']):
                viable.append((summary['mean_r1_difference'],-summary['mean_ne_difference_m'],-summary['broken_group_rate'],-scale,threshold,i))
            print(json.dumps(dict(fit_choices=len(choices),scale=scale,threshold=threshold)),flush=True)
    selected_index=max(viable)[-1] if viable else None
    selected=choices[selected_index]['parameters'] if selected_index is not None else dict(scale=0.,threshold=1.)
    immutable_json(dict(temperature=temp_report,choices=choices,selected_index=selected_index,selected_parameters=selected,selection='FIT targets only'),a.output/'fit_decision.json')
    # No alternative choice is tested after this unique held-out verification.
    stats=evaluate_gate(verify,selected,temperature);risk=verify_gate(stats)
    enabled=bool(selected_index is not None and temp_report['sufficient'] and risk['passed'] and verification['checkpoint_eligible'])
    result=dict(status='revision_candidate_requires_real_closed_loop_pilot',gate_final=False,enabled=enabled,
                scale=selected['scale'] if enabled else 0.,threshold=selected['threshold'] if enabled else 1.,temperature=temperature,
                checkpoint_sha256=sha(checkpoint),fit_selected_parameters=selected,temperature_report=temp_report,
                verification=stats,verify_gate=risk,fit_groups=len({tuple(p['group']) for p in fit}),verify_groups=len({tuple(p['group']) for p in verify}),
                observation_source=source_signature['observation_source'],no_verify_reselection=True,benchmark_used=False,
                analysis_signature_sha256=sha(a.output/'analysis_signature.json'),confirmation_manifest_sha256=sha(a.confirmation/'manifest.json'),packet_manifest_sha256=sha(a.output/'packets/manifest.json'),
                training_verification_sha256=sha(BASE/'revision_v1/training/artifact_verification.json'),seconds=time.perf_counter()-started)
    save(result,a.output/'candidate_calibration.json');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
