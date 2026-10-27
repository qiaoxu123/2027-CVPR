"""Verify full training, immutable checkpoints and logged curves before calibration."""
import json
import math
from pathlib import Path
import torch
from .paths import ROOT,A0_SHA,sha,save
from .training_protocol import TRAIN_CONFIG,validate_complete_cache,selection_key


def validate_training_history(history,train_views,dev_views):
    if len(history)!=25 or [h['epoch'] for h in history]!=list(range(1,26)):
        raise ValueError('All25 sequential epochs are required')
    for row in history:
        refine=row['epoch']>20
        if row['stage']!=('refinement' if refine else 'ranking') or row['stage_epoch']!=(row['epoch']-20 if refine else row['epoch']):
            raise ValueError('Training stage/epoch schedule changed')
        if row['train']['views']!=train_views or row['dev']['views']!=dev_views:
            raise ValueError('An epoch did not process the full train/dev views')
        if row['train']['optimizer_updates']<0:
            raise ValueError('Invalid optimizer update count')


def verify_curves(history,directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from .train import write_record_scalars
    expected={}
    class Collector:
        def add_scalar(self,tag,value,step):expected.setdefault(tag,{})[step]=float(value)
    for record in history:write_record_scalars(record,Collector())
    events=EventAccumulator(str(directory/'tensorboard'),size_guidance={'scalars':0}).Reload()
    differences=[]
    for tag,values in expected.items():
        actual={e.step:e.value for e in events.Scalars(tag)}
        if set(actual)!=set(values):raise ValueError(f'TensorBoard steps differ: {tag}')
        for step,value in values.items():
            differences.append(abs(actual[step]-value))
            if not math.isclose(actual[step],value,rel_tol=1e-6,abs_tol=1e-5):
                raise ValueError(f'TensorBoard/history mismatch: {tag} step{step}')
    return dict(tags=len(expected),scalar_values=len(differences),max_abs_difference=max(differences,default=0.))


def verify_training_artifacts(directory,cache,*,hash_cache_rows=True):
    from .train import runtime_hashes,training_hashes
    from .prepare import cache_signature
    directory,cache=Path(directory),Path(cache)
    marker=json.loads((directory/'training_complete.json').read_text())
    if marker['status']!='full20plus5_training_complete_calibration_pending' or marker['rank_epochs']!=20 or marker['refine_epochs']!=5:
        raise ValueError('Full20+5 training completion marker required')
    if cache_signature()!=json.loads((cache/'signature.json').read_text()):
        raise ValueError('Frozen data/feature-extraction provenance changed')
    source=json.loads((ROOT/'runs/a2_waldo_landmark_20261027/source_manifest.json').read_text())
    manifest=validate_complete_cache(cache,source,hash_rows=hash_cache_rows)
    counts={p:sum(e['split']==p for e in manifest['entries']) for p in ('train','dev','calibration')}
    history=json.loads((directory/'history.json').read_text())
    validate_training_history(history,counts['train'],counts['dev'])
    if sha(directory/'history.json')!=marker['history_sha256']:
        raise ValueError('Completed training history changed')
    protocol=json.loads((directory/'protocol.json').read_text())
    if protocol['config']!=TRAIN_CONFIG or protocol['cache_manifest_sha256']!=sha(cache/'manifest.json'):
        raise ValueError('Training recipe or full cache manifest changed')
    stage_updates=0
    for record in history:
        epoch=record['epoch']
        if epoch==21:stage_updates=0
        stage_updates+=record['train']['optimizer_updates']
        state=torch.load(directory/f'epoch_{epoch:02d}.pt',map_location='cpu',weights_only=False)
        if (state['epoch']!=epoch or state['history']!=history[:epoch] or state['a0_sha256']!=A0_SHA
                or state['config']!=TRAIN_CONFIG or state['runtime_code_sha256']!=runtime_hashes()
                or state['training_modules_sha256']!=training_hashes()
                or state['cache_signature_sha256']!=manifest['signature_sha256']):
            raise ValueError(f'Epoch{epoch} checkpoint metadata/provenance mismatch')
        counter=max((int(s['step']) for s in state['optimizer']['state'].values() if 'step' in s),default=0)
        if counter!=stage_updates:raise ValueError(f'Epoch{epoch} optimizer update counter mismatch')
    selected=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    epoch=selected['epoch']
    if epoch not in range(21,26) or marker['chosen_epoch']!=epoch:
        raise ValueError('Final checkpoint must be selected from the five completed refinement epochs')
    expected=max(history[20:],key=lambda h:selection_key(h['dev'],refine=True))['epoch']
    if epoch!=expected:raise ValueError('Final checkpoint does not follow the predeclared TRAIN-dev selection')
    rank_epoch=max(history[:20],key=lambda h:selection_key(h['dev'],refine=False))['epoch']
    if sha(directory/'best_rank.pt')!=sha(directory/f'epoch_{rank_epoch:02d}.pt'):
        raise ValueError('Ranking checkpoint does not follow TRAIN-dev selection')
    if (sha(directory/'best.pt')!=marker['checkpoint_sha256']
            or sha(directory/'best.pt')!=sha(directory/f'epoch_{epoch:02d}.pt')):
        raise ValueError('Selected checkpoint no longer matches its completed epoch')
    curves=verify_curves(history,directory)
    report=dict(status='full_training_artifacts_verified',rank_epochs=20,refine_epochs=5,counts=counts,
                chosen_epoch=epoch,checkpoint_sha256=marker['checkpoint_sha256'],checkpoint_eligible=selected['checkpoint_eligible'],
                cache_signature_sha256=manifest['signature_sha256'],cache_rows_hashed=hash_cache_rows,
                curves=curves,all_epoch_optimizer_counters_verified=True)
    save(report,directory/'artifact_verification.json')
    return selected,report
