"""Immutable ORIGINAL TRAIN/dev risk cache; no confirmation-label consumption."""
import json
from pathlib import Path
import numpy as np
import torch
from .paths import ROOT,sha
from .collected_snapshot import BASE
from .risk_probe import FEATURE_NAMES,WEIGHTS
from .direct_risk_head import outcome_classes


def validate_arrays(data,identities,count):
    shapes=dict(features=(count,len(FEATURE_NAMES)),outcomes=(count,len(WEIGHTS)),before_ne=(count,),after_ne=(count,len(WEIGHTS)))
    if set(data)!=set(shapes) or len(identities)!=count:raise ValueError('Incomplete risk cache schema/population')
    if any(data[k].shape!=s or not np.isfinite(data[k]).all() for k,s in shapes.items()):raise ValueError('Invalid risk cache shapes/values')
    if (data['before_ne']<0).any() or (data['after_ne']<0).any():raise ValueError('Negative private distance')
    gain=(data['before_ne'][:,None]>20)&(data['after_ne']<=20);harm=(data['before_ne'][:,None]<=20)&(data['after_ne']>20)
    if not np.array_equal(gain.astype(np.int8)-harm.astype(np.int8),data['outcomes']):raise ValueError('Risk labels differ from actual fixed actions')
    if len({(r['episode'],r['step']) for r in identities})!=count:raise ValueError('Duplicate risk observation')


def normalization(train):
    x=train['features'].astype(np.float64);classes=outcome_classes(torch.as_tensor(train['outcomes'][:,-1])).numpy()
    counts=np.bincount(classes,minlength=3)
    if (counts<=0).any():raise ValueError('TRAIN requires all three outcome classes')
    return x.mean(0).astype(np.float32),x.std(0).clip(1e-6).astype(np.float32),(len(x)/(3*counts)).astype(np.float32)


def load_original_risk_data(directory=BASE/'revision_v2_probe'):
    directory=Path(directory);marker=json.loads((directory/'complete.json').read_text());signature=json.loads((directory/'signature.json').read_text())
    if marker['status']!='complete_original_TRAIN_dev_risk_probe_extraction' or marker['signature_sha256']!=sha(directory/'signature.json'):
        raise ValueError('Complete risk inference marker missing/changed')
    if signature['features']!=list(FEATURE_NAMES) or signature['weights']!=list(WEIGHTS):raise ValueError('Risk feature/action schema changed')
    if signature['next_confirmation_reserve_sha256']!=sha(BASE/'revision_v2_confirmation_reserve.json') or signature['cache_manifest_sha256']!=sha(BASE/'collected_snapshot/manifest.json'):
        raise ValueError('Frozen TRAIN/independent reserve changed')
    for name,digest in signature['code'].items():
        if Path(name).name!=name or sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)!=digest:raise ValueError('Frozen extraction dependency changed')
    if signature['extract_code_sha256']!=sha(ROOT/'scripts/a2_risk_probe_extract.py'):raise ValueError('Risk extractor code changed')
    rank=BASE/'revision_v1/training';checkpoint=rank/'best.pt'
    if signature['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Frozen relation checkpoint changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for name,digest in state['code_sha256'].items():
        if Path(name).name!=name or sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)!=digest:raise ValueError('Original relation training dependency changed')
    parent_marker=json.loads((rank/'training_complete.json').read_text());parent_proof=json.loads((rank/'artifact_verification.json').read_text())
    parent_history=json.loads((rank/'history.json').read_text());old_calibration=json.loads((BASE/'revision_v1/calibration/candidate_calibration.json').read_text())
    if (parent_marker['rank_epochs']!=20 or parent_marker['refine_epochs']!=5 or len(parent_history)!=25
        or [r['epoch'] for r in parent_history]!=list(range(1,26)) or parent_marker['history_sha256']!=sha(rank/'history.json')
        or parent_marker['checkpoint_sha256']!=signature['checkpoint_sha256'] or parent_proof['checkpoint_sha256']!=signature['checkpoint_sha256']
        or not parent_proof['all_optimizer_steps_verified'] or old_calibration['training_verification_sha256']!=sha(rank/'artifact_verification.json')):
        raise ValueError('Original completed relation20+5 proof changed')
    data={};identities={};groups={}
    for partition,count in (('train',31415),('dev',4618)):
        expected=marker['partitions'][partition]
        if expected['views']!=count or expected['npz_sha256']!=sha(directory/f'{partition}.npz') or expected['identities_sha256']!=sha(directory/f'{partition}_identities.json'):
            raise ValueError('Corrupt/incomplete risk cache source')
        with np.load(directory/f'{partition}.npz') as f:data[partition]={k:f[k].copy() for k in f.files}
        identities[partition]=json.loads((directory/f'{partition}_identities.json').read_text())
        validate_arrays(data[partition],identities[partition],count)
        groups[partition]={(r['map_name'],r['object_id']) for r in identities[partition]}
    reserved={(r['map_name'],r['object_id']) for r in json.loads((BASE/'revision_v2_confirmation_reserve.json').read_text())['targets']}
    if len(reserved)!=256 or groups['train']&groups['dev'] or reserved&(groups['train']|groups['dev']):raise ValueError('Risk TRAIN/dev/confirmation targets overlap')
    lineage=dict(source_complete_sha256=sha(directory/'complete.json'),source_signature_sha256=sha(directory/'signature.json'),
        rank_checkpoint_sha256=signature['checkpoint_sha256'],rank_training_proof_sha256=sha(rank/'artifact_verification.json'),
        rank_training_marker_sha256=sha(rank/'training_complete.json'),next_reserve_sha256=sha(BASE/'revision_v2_confirmation_reserve.json'))
    return data,identities,lineage
