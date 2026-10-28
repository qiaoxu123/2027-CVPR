"""Verify completed revision training before independent confirmation/deployment."""
import json,math
from pathlib import Path
import torch
from .paths import ROOT,A0_SHA,sha,save
from .collected_snapshot import BASE,verify_cache_origin,cache_source
from .training_protocol import validate_complete_cache
from .revision_train import CONFIG,code_hashes,revision_selection
from .artifact_protocol import verify_curves


def verify_revision_history(history,train_views,dev_views):
    if len(history)!=25 or [r['epoch'] for r in history]!=list(range(1,26)):raise ValueError('Revision requires all25 sequential epochs')
    for row in history:
        refine=row['epoch']>20
        if row['stage']!=('refinement' if refine else 'ranking') or row['stage_epoch']!=(row['epoch']-20 if refine else row['epoch']):raise ValueError('Revision stage schedule changed')
        for split,count in (('train',train_views),('dev',dev_views)):
            m=row[split]
            if m['views']!=count or not math.isfinite(m['loss']):raise ValueError('Revision population/loss invalid')
            expected_updates=math.ceil(count/CONFIG['batch_size']) if split=='train' else 0
            if m['optimizer_updates']!=expected_updates:raise ValueError('Revision epoch optimizer work incomplete or validation updated weights')
            if m['map_correct']!=m['a0_correct']+m['corrected']-m['broken']:raise ValueError('Correction/break accounting inconsistent')
            if not 0<=m['proposal_correct']<=m['proposal_available_views']<=count:raise ValueError('Proposal population invalid')
            if not math.isclose(m['map_r1'],m['map_correct']/count,abs_tol=1e-10):raise ValueError('Map metric denominator changed')


def verify_revision_training(training=BASE/'revision_v1/training',cache=BASE/'collected_snapshot'):
    training,cache=Path(training),Path(cache)
    marker=json.loads((training/'training_complete.json').read_text())
    if marker['status']!='revision_full20plus5_complete_independent_confirmation_pending' or marker['rank_epochs']!=20 or marker['refine_epochs']!=5:
        raise ValueError('Completed revision20+5 marker required')
    verify_cache_origin(cache);manifest=validate_complete_cache(cache,cache_source(cache),hash_rows=True)
    protocol=json.loads((training/'protocol.json').read_text());hashes=code_hashes()
    expected=dict(code_sha256=hashes,config=CONFIG,cache_manifest_sha256=sha(cache/'manifest.json'),
                  cache_signature_sha256=manifest['signature_sha256'],reserve_sha256=sha(BASE/'revision_confirmation_reserve.json'),a0_sha256=A0_SHA)
    if any(protocol[k]!=v for k,v in expected.items()):raise ValueError('Revision protocol/code/data provenance changed')
    actual_counts={p:sum(e['split']==p for e in manifest['entries']) for p in ('train','dev')}
    if any(protocol[p+'_views']!=n for p,n in actual_counts.items()):raise ValueError('Revision protocol population differs from frozen manifest')
    h=json.loads((training/'history.json').read_text())
    verify_revision_history(h,protocol['train_views'],protocol['dev_views'])
    if sha(training/'history.json')!=marker['history_sha256']:raise ValueError('Completed revision history changed')
    updates=0;previous_weights=None
    for record in h:
        e=record['epoch']
        if e==21:
            updates=0
            previous_weights=torch.load(training/'best_rank.pt',map_location='cpu',weights_only=False)['model']
        updates+=record['train']['optimizer_updates']
        state=torch.load(training/f'epoch_{e:02d}.pt',map_location='cpu',weights_only=False)
        if any(state[k]!=v for k,v in expected.items()) or state['epoch']!=e or state['history']!=h[:e]:raise ValueError(f'Revision epoch{e} metadata changed')
        if any(not torch.isfinite(t).all() for t in state['model'].values()):raise ValueError('Nonfinite completed revision weights')
        if previous_weights is not None and all(torch.equal(t,previous_weights[k]) for k,t in state['model'].items()):raise ValueError('No revision weight changed during an epoch')
        previous_weights=state['model']
        counter=max((int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v),default=0)
        if counter!=updates:raise ValueError('Revision optimizer counter mismatch')
    for refine,alias,records in ((False,'best_rank.pt',h[:20]),(True,'best.pt',h[20:])):
        chosen=max(records,key=lambda r:revision_selection(r['dev'],refine=refine))['epoch']
        if sha(training/alias)!=sha(training/f'epoch_{chosen:02d}.pt'):raise ValueError('Revision best alias violates frozen selection')
    selected=torch.load(training/'best.pt',map_location='cpu',weights_only=False)
    if marker['selected_epoch']!=selected['epoch'] or sha(training/'best.pt')!=marker['checkpoint_sha256']:raise ValueError('Selected revision checkpoint changed')
    curves=verify_curves(h,training)
    report=dict(status='revision_all25_artifacts_verified',checkpoint_sha256=marker['checkpoint_sha256'],selected_epoch=selected['epoch'],
                checkpoint_eligible=selected['checkpoint_eligible'],dev=selected['history'][-1]['dev'],curves=curves,
                all_optimizer_steps_verified=True,source_signature=expected,confirmation_used=False,benchmark_used=False)
    save(report,training/'artifact_verification.json')
    return selected,report
