"""Frozen-FIT-authorized V3 inference on old130 A0 observation records."""
import fcntl,json,time
from pathlib import Path
import torch
from .paths import ROOT,sha,save
from .collected_snapshot import BASE
from .relation_v3_calibration_data import FreshConfirmation,calibration_code_hashes,load_frozen_choice
from .relation_v3_confirmation_protocol import immutable,verify_opaque_seal
from .relation_v3_confirmation import archive_partial
from .relation_v3_model import RelationV3Network
from .relation_v3_runtime import public_network_inputs,runtime_forward
from .relation_v3_packet import make_packet
from .relation_v3_train import require_idle_gpu
from .calibration_protocol import calibration_fold
from .train import atomic_checkpoint


def read_sealed_original(directory,item,query,signature,*,authorized=False):
    if not authorized:raise ValueError('Frozen nonempty FIT choice required before sealed tensor read')
    name=item['filename']
    if (name!=f"view_{query['index']}_{item['step']}.pt" or Path(name).name!=name or item['episode']!=query['index']
        or query['fold']!='verify' or calibration_fold(query['map_name'],query['object_id'])!='verify'):
        raise ValueError('Sealed source query/fold changed')
    path=Path(directory)/name
    if sha(path)!=item['raw_sha256']:raise ValueError('Original sealed row changed')
    row=torch.load(path,map_location='cpu',weights_only=False)
    if (row['signature_sha256']!=signature or row['episode']!=item['episode'] or row['step']!=item['step']
        or row['fold']!='verify' or row['target_instance']!=[query['map_name'],query['object_id']]):raise ValueError('Sealed raw header differs from fixed source')
    return row


def check_replay_population(rows,expected,target_indices):
    identity=lambda e:(e['filename'],e['episode'],e['step'],e['original_raw_sha256'])
    actual=[identity(e) for e in rows]
    original=[(e['filename'],e['episode'],e['step'],e['raw_sha256']) for e in expected]
    if (actual!=original or len(set(actual))!=len(actual) or {e['episode'] for e in rows}!=set(target_indices)):
        raise ValueError('Sealed replay population is partial, duplicate or rebound')


def signature_for(source):
    root=source.root;load_frozen_choice(root,source.checkpoint_sha256,source.roles_sha256)
    return dict(checkpoint_sha256=source.checkpoint_sha256,roles_sha256=source.roles_sha256,
        training_verification_sha256=sha(root/'training/artifact_verification.json'),
        analysis_signature_sha256=sha(root/'calibration/analysis_signature.json'),fit_decision_sha256=sha(root/'calibration/fit_decision.json'),
        fit_decision_seal_sha256=sha(root/'calibration/fit_decision_seal.json'),sealed_verify_sha256=sha(root/'sealed_verify130.json'),
        original_signature_sha256=sha(BASE/'revision_v2/confirmation/signature.json'),implementation=calibration_code_hashes(),
        scope='Cached real originalA0 observations; V3 public-only inference AFTER unique frozen FIT choice; no GT input')


@torch.no_grad()
def infer_row(raw,model,signature,original_sha,*,device='cuda'):
    network=public_network_inputs(raw['public'],raw['meta'])
    network={k:v.to(device) for k,v in network.items()};spatial={k:v.to(device) for k,v in raw['public']['spatial'].items()}
    with torch.autocast(torch.device(device).type,enabled=False):
        out=model(**{k:v[None] for k,v in network.items()})
        _,_,risk=runtime_forward(spatial,out,model,dict(enabled=False))
    # Private coordinates are copied only AFTER the complete public forward;
    # the original A0 rollout ended before these source labels were created.
    return dict(public=dict(network={k:v.detach().cpu().clone() for k,v in network.items()},
                            spatial={k:v.detach().cpu().clone() for k,v in spatial.items()}),
        prediction={k:v[0].detach().cpu().clone() for k,v in out.items()},risk={k:v[0].detach().cpu().clone() for k,v in risk.items()},
        meta=raw['meta'],observation=raw.get('observation'),labels=dict(target_xy=raw['labels']['target_xy'].detach().cpu().clone()),
        target_instance=list(raw['target_instance']),fold=raw.get('fold',calibration_fold(*raw['target_instance'])),
        episode=raw['episode'],step=raw['step'],signature_sha256=signature,original_raw_sha256=original_sha)


def numerical_probe(source,model):
    chosen=[];targets=set()
    for family in ('teacher_snapshot','revision_v1_A0_closedloop','revision_v2_A0_closedloop'):
        count=0
        for e in source.roles['rows']:
            key=(e['map_name'],e['object_id'])
            if e['role']!='train' or e['source_kind']!=family or key in targets:continue
            chosen.append(e);targets.add(key);count+=1
            if count==2:break
        if count!=2:raise ValueError('All three TRAIN families required for numerical probe')
    for e in chosen:
        path=Path(e['directory'])/e['filename']
        if sha(path)!=e['sha256']:raise ValueError('TRAIN numerical-probe source changed')
        raw=torch.load(path,map_location='cpu',weights_only=False)
        if raw['target_instance']!=[e['map_name'],e['object_id']]:raise ValueError('TRAIN probe identity changed')
        result=infer_row(raw,model,'TRAIN_numerical_probe_only',e['sha256'])
        result['fold']=calibration_fold(*result['target_instance'])
        make_packet(result,source.model)  # No scores/NE/gains are reported or fitted.
    return dict(scope='TRAIN-only numerical integrity, not performance or independent confirmation',cases=6,
        sources=[{k:e[k] for k in ('source_kind','filename','sha256')} for e in chosen],passed=True)


def replay_manifest(source):
    root=source.root;directory=root/'sealed_replay';expected_signature=signature_for(source)
    signature=json.loads((directory/'signature.json').read_text())
    if signature!=expected_signature:raise ValueError('Sealed replay signature changed')
    digest=sha(directory/'signature.json');manifest=json.loads((directory/'manifest.json').read_text())
    seal=json.loads((root/'sealed_verify130.json').read_text());queries={e['index']:e for e in seal['targets']}
    if manifest['status']!='v3_sealed130_public_inference_replay_complete' or manifest['signature_sha256']!=digest or manifest['target_groups']!=130:
        raise ValueError('Complete sealed130 V3 replay required')
    check_replay_population(manifest['entries'],seal['rows'],set(queries))
    for e in manifest['entries']:
        path=directory/e['filename'];sidecar=path.with_suffix('.sha.json')
        if sha(path)!=e['sha256'] or sha(sidecar)!=e['sidecar_sha256']:raise ValueError('V3 replay row commit changed')
        bound=json.loads(sidecar.read_text())
        if bound!=dict(row_sha256=sha(path),original_raw_sha256=e['original_raw_sha256'],signature_sha256=digest):raise ValueError('V3 replay row sidecar changed')
    return manifest,queries,digest


def run_replay():
    source=FreshConfirmation();root=source.root;signature=signature_for(source)
    seal=json.loads((root/'sealed_verify130.json').read_text());verify_opaque_seal(seal)
    directory=root/'sealed_replay';directory.mkdir(parents=True,exist_ok=True);immutable(signature,directory/'signature.json');digest=sha(directory/'signature.json')
    if (directory/'manifest.json').exists():replay_manifest(source);print('Retained complete frozen-choice sealed replay',flush=True);return
    require_idle_gpu();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=RelationV3Network(**source.state['model_config']);model.load_state_dict(source.state['model'],strict=True)
    require_idle_gpu();model=model.cuda().eval().requires_grad_(False)
    probe=numerical_probe(source,model);immutable(dict(signature_sha256=digest,**probe),directory/'TRAIN_numerical_probe.json')
    queries={e['index']:e for e in seal['targets']};entries=[];start=time.perf_counter()
    for number,e in enumerate(seal['rows'],1):
        require_idle_gpu(allow_self=True) if number%100==1 else None
        path=directory/e['filename'];sidecar=path.with_suffix('.sha.json')
        if path.exists() and sidecar.exists():
            bound=json.loads(sidecar.read_text())
            if bound!=dict(row_sha256=sha(path),original_raw_sha256=e['raw_sha256'],signature_sha256=digest):raise ValueError('Existing replay commit changed')
            row=torch.load(path,map_location='cpu',weights_only=False)
        else:
            raw=read_sealed_original(BASE/'revision_v2/confirmation',e,queries[e['episode']],signature['original_signature_sha256'],authorized=True)
            row=infer_row(raw,model,digest,e['raw_sha256']);archive_partial(path,directory);atomic_checkpoint(row,path)
            save(dict(row_sha256=sha(path),original_raw_sha256=e['raw_sha256'],signature_sha256=digest),sidecar)
        if (row['signature_sha256']!=digest or row['original_raw_sha256']!=e['raw_sha256'] or row['episode']!=e['episode'] or row['step']!=e['step']
            or row['fold']!='verify' or row['target_instance']!=[queries[e['episode']]['map_name'],queries[e['episode']]['object_id']]):raise ValueError('Replayed target/source header changed')
        entries.append(dict(filename=e['filename'],sha256=sha(path),sidecar_sha256=sha(sidecar),original_raw_sha256=e['raw_sha256'],episode=e['episode'],step=e['step']))
        if number%100==0:print(json.dumps(dict(replayed=number,total=1034,seconds=time.perf_counter()-start)),flush=True)
    check_replay_population(entries,seal['rows'],set(queries))
    immutable(dict(status='v3_sealed130_public_inference_replay_complete',signature_sha256=digest,target_groups=130,
        entries=entries,no_comparative_performance_aggregate=True),directory/'manifest.json')
    replay_manifest(source);print(json.dumps(dict(status='sealed130_inference_complete_VERIFY_aggregate_pending',rows=len(entries))),flush=True)


def load_replayed_packets(source):
    manifest,queries,digest=replay_manifest(source);directory=source.root/'sealed_replay'
    output=source.root/'calibration/packets_sealed_verify';output.mkdir(parents=True,exist_ok=True)
    signature=dict(replay_manifest_sha256=sha(directory/'manifest.json'),replay_signature_sha256=digest,
        fit_decision_sha256=sha(source.root/'calibration/fit_decision.json'),checkpoint_sha256=source.checkpoint_sha256,
        implementation=calibration_code_hashes())
    immutable(signature,output/'signature.json');packet_digest=sha(output/'signature.json');packets=[];entries=[]
    for e in manifest['entries']:
        path=output/e['filename'];sidecar=path.with_suffix('.sha.json')
        if path.exists() and sidecar.exists():
            bound=json.loads(sidecar.read_text())
            if bound!=dict(packet_sha256=sha(path),source_row_sha256=e['sha256'],signature_sha256=packet_digest):raise ValueError('Derived sealed packet changed')
            packet=torch.load(path,map_location='cpu',weights_only=False)
        else:
            row=torch.load(directory/e['filename'],map_location='cpu',weights_only=False);packet=make_packet(row,source.model)
            packet.update(source_row_sha256=e['sha256'],signature_sha256=packet_digest);atomic_checkpoint(packet,path)
            save(dict(packet_sha256=sha(path),source_row_sha256=e['sha256'],signature_sha256=packet_digest),sidecar)
        query=queries[e['episode']]
        if (packet['group']!=[query['map_name'],query['object_id']] or packet['fold']!='verify' or packet['episode']!=e['episode'] or packet['step']!=e['step']
            or packet['source_row_sha256']!=e['sha256'] or packet['signature_sha256']!=packet_digest):raise ValueError('Derived sealed packet identity changed')
        packets.append(packet);entries.append(dict(filename=e['filename'],sha256=sha(path),sidecar_sha256=sha(sidecar),source_row_sha256=e['sha256']))
    if len(packets)!=1034 or len({tuple(p['group']) for p in packets})!=130:raise ValueError('Incomplete sealed target/observation packets')
    immutable(dict(signature_sha256=packet_digest,target_groups=130,entries=entries),output/'manifest.json')
    return packets


def main():
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run_replay()


if __name__=='__main__':main()
