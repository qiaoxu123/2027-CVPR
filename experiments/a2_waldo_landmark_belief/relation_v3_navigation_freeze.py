"""Freeze V3 after unique calibration and a complete, genuinely changed pilot."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import torch
from .paths import CHECKPOINT,A0_SHA,sha,verify_frozen_assets
from .collected_snapshot import BASE
from .revision_freeze import evaluation_data_signature
from .relation_v3_confirmation_protocol import immutable,verify_opaque_seal
from .relation_v3_navigation_admission import load_candidate
from .relation_v3_navigation_sources import code_hashes,require_cpu_gates,FULL_COUNTS


def external_assets():
    from huggingface_hub import snapshot_download
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    references=((state['a3_args']['text_backbone'],state['config']['frozen_backbones']['bert_revision']),
        (state['protocol']['backbone'],state['protocol']['backbone_revision']))
    encoders={}
    for name,revision in references:
        snapshot=Path(snapshot_download(name,revision=revision,local_files_only=True))
        files=sorted(p for p in snapshot.rglob('*') if p.is_file())
        if not files or not any(p.suffix in ('.bin','.safetensors') for p in files):raise ValueError('Actual frozen encoder weights unavailable')
        encoders[name]=dict(revision=revision,files={str(p):sha(p) for p in files})
    return dict(original=verify_frozen_assets(),encoders=encoders,data=evaluation_data_signature())


def require_assets(root,*,create=False):
    root=Path(root);path=root/'navigation_assets.json';current=external_assets()
    if create:immutable(current,path)
    if json.loads(path.read_text())!=current:raise ValueError('Original dependencies, encoders or evaluation data changed')
    return current


def final_parameters(candidate,root):
    return dict(candidate,status='v3_parameters_frozen_after_real_pilot',gate_final=True,
        pilot_report_sha256=sha(Path(root)/'pilot/report.json'))


def require_pilot(pilot,checkpoint_sha256,candidate_sha256,implementation):
    if (not pilot['decision']['passed'] or pilot['decision']['episodes']!=64 or pilot['benchmark_used']
        or pilot['checkpoint_sha256']!=checkpoint_sha256 or pilot['calibration_candidate_sha256']!=candidate_sha256
        or pilot['code_sha256']!=implementation):raise ValueError('Complete safe changed64 real pilot with matching sources required')


def check_bound_files(bound):
    for name,digest in bound.items():
        if sha(name)!=digest:raise ValueError('Frozen navigation artifact changed: '+name)


def bound_paths(root):
    root=Path(root)
    paths={root/n for n in ('final_calibration.json','navigation_assets.json','navigation_cpu_gates.json',
        'navigation_cpu_tests.log','navigation_protocol.json','navigation_pilot_population.json','roles.json',
        'confirmation_population.json','confirmation_protocol.json','sealed_verify130.json','posttraining_cpu_gates.json',
        'training/best.pt','training/training_complete.json','training/artifact_verification.json','training/curve_manifest.json')}
    for directory in ('confirmation','calibration','sealed_replay','pilot'):
        paths.update(p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.json','.pt','.png','.jsonl')
            and not any('archive' in part for part in p.relative_to(root).parts))
    return sorted(paths)


def validate_frozen(root=BASE/'revision_v3'):
    root=Path(root);require_cpu_gates(root)
    frozen=json.loads((root/'frozen.json').read_text())
    if (frozen['status']!='v3_frozen_after_independent_calibration_and_real_pilot' or not frozen['enabled']
        or frozen['episode_counts']!=FULL_COUNTS or frozen['no_val_unseen_selection'] is not True
        or frozen['code_sha256']!=code_hashes() or frozen['a0_sha256']!=A0_SHA or sha(CHECKPOINT)!=A0_SHA
        or frozen['checkpoint_sha256']!=sha(root/'training/best.pt')):raise ValueError('Final V3 freeze invalid or changed')
    check_bound_files(frozen['bound_files'])
    # Bound every consumed calibration artifact, not merely the report headers.
    if set(frozen['bound_files'])!={str(p.resolve()) for p in bound_paths(root)}:raise ValueError('Frozen navigation artifact inventory changed')
    require_assets(root)
    verify_opaque_seal(json.loads((root/'sealed_verify130.json').read_text()))
    candidate,_=load_candidate(root)
    pilot=json.loads((root/'pilot/report.json').read_text())
    require_pilot(pilot,frozen['checkpoint_sha256'],sha(root/'calibration/candidate_calibration.json'),code_hashes())
    if json.loads((root/'final_calibration.json').read_text())!=final_parameters(candidate,root):
        raise ValueError('Final parameters differ from unique calibrated candidate')
    return frozen


def freeze(root=BASE/'revision_v3'):
    root=Path(root);require_cpu_gates(root)
    if (root/'frozen.json').exists():return validate_frozen(root)
    candidate,_=load_candidate(root);require_assets(root)
    from .relation_v3_navigation import verify_pilot_outputs
    pilot=verify_pilot_outputs()
    checkpoint=sha(root/'training/best.pt');candidate_digest=sha(root/'calibration/candidate_calibration.json')
    require_pilot(pilot,checkpoint,candidate_digest,code_hashes())
    verify_opaque_seal(json.loads((root/'sealed_verify130.json').read_text()))
    immutable(final_parameters(candidate,root),root/'final_calibration.json')
    frozen=dict(status='v3_frozen_after_independent_calibration_and_real_pilot',enabled=True,
        checkpoint_sha256=checkpoint,a0_sha256=A0_SHA,code_sha256=code_hashes(),
        bound_files={str(p.resolve()):sha(p) for p in bound_paths(root)},episode_counts=FULL_COUNTS,
        no_val_unseen_selection=True,model_selected_using='Complete TRAIN-internal dev; one independent FIT choice and VERIFY aggregate; fixed real pilot',
        frozen_at_utc=datetime.now(timezone.utc).isoformat())
    immutable(frozen,root/'frozen.json');validate_frozen(root);return frozen


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();print(json.dumps(freeze()),flush=True)
