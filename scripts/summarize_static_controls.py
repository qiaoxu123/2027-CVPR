"""Collect immutable configurations, selected checkpoints, curves and paired outcomes."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from graphnav.checkpointing import file_sha256

ROOT=Path(__file__).resolve().parents[1]


def paired(original, candidate, key, map_key):
    a=[json.loads(line) for line in Path(original).read_text().splitlines()]
    b=[json.loads(line) for line in Path(candidate).read_text().splitlines()]
    if len(a)!=len(b):raise ValueError('paired size mismatch')
    for x,y in zip(a,b):
        if x['index']!=y['index']:raise ValueError('paired index mismatch')
    def value(row):
        for part in key.split('.'):
            row=row[part]
        return float(row)
    diff=np.array([value(y)-value(x) for x,y in zip(a,b)])
    maps=np.array([x[map_key] for x in a]);unique=np.unique(maps)
    rng=np.random.default_rng(0)
    samples=[]
    for _ in range(2000):
        indices=np.concatenate([np.flatnonzero(maps==m) for m in rng.choice(unique,len(unique),replace=True)])
        samples.append(diff[indices].mean())
    return dict(n=len(a),delta=float(diff.mean()),improved=int((diff>0).sum()),
        regressed=int((diff<0).sum()),map_bootstrap_95ci=np.quantile(samples,[.025,.975]).tolist())


def main():
    source=ROOT/'runs/static_final_analysis_20261010'
    destination=ROOT/'reports/a3_relation_gat_20261010/final'
    destination.mkdir(exist_ok=False)
    static=json.loads((source/'static_seen/metrics.json').read_text())
    factorial=json.loads((source/'direction_factorial.json').read_text())
    for filename in ('direction_factorial.json','relation_predicates.json','status.json'):
        shutil.copy2(source/filename,destination/filename)
    summary=dict(models={},paired_static={},paired_closedloop={},notes=[
        'Static uses deduplicated 2426 Seen queries; closed-loop uses same 128 episode SHA.',
        'Selected checkpoints use Seen R1 then NE. One Seed per E2/E3: descriptive evidence, no general superiority claim.',
        'Direction test uses 237 synthetic controlled pairs. Predicate diagnostics are post-forward geometry tests, not a trained classifier.',
        'Raw static/closedloop predictions and checkpoints stay in local runs; all paths and SHA are recorded.'])
    rows=[]
    for name,metrics in static['models'].items():
        state=torch.load(metrics['checkpoint'],map_location='cpu',weights_only=True)
        run=Path(metrics['checkpoint']).parent
        history=json.loads((run/'history.json').read_text()) if (run/'history.json').exists() else state['history']
        curve=destination/(name+'_training.json');curve.write_text(json.dumps(history,indent=2))
        if (run/'training.csv').exists():shutil.copy2(run/'training.csv',destination/(name+'_training.csv'))
        closed=source/'seen128'/name/'metrics.json'
        if name in ('A3_seed0','E1_seed0'):
            closed=ROOT/'reports/a3_relation_gat_20261010'/('seed0_epoch5_seen128.json' if name=='A3_seed0' else 'E1_best_seen128.json')
        navigation=json.loads(closed.read_text())
        shutil.copy2(closed,destination/(name+'_seen128.json'))
        direction=factorial['models'][name]
        summary['models'][name]=dict(static=metrics,config=state['args'],provenance=state.get('provenance'),
            fingerprint=state.get('input_fingerprint'),closedloop=navigation['metrics'],
            episode_sha256=navigation['episode_manifest_sha256'],direction=direction,
            raw_static_predictions=str(source/'static_seen'/(name+'_predictions.jsonl')),
            raw_closedloop_predictions=str(Path(navigation['args']['output'])/'predictions.jsonl'))
        rows.append(dict(model=name,epoch=metrics['epoch'],R1=metrics['R@1/20m'],R4=metrics['R@4/20m'],R8=metrics['R@8/20m'],
            NE=metrics['NE_m'],direction_original=direction['original_direction_accuracy'],
            direction_swapped=direction['full_swap']['inverse_direction_accuracy'],flip=direction['full_swap']['flipped_fraction'],
            both_correct=direction['full_swap']['both_directions_correct'],word_flip=direction['word_only']['flipped_fraction'],
            SR=navigation['metrics']['SR'],SPL=navigation['metrics']['SPL'],OSR=navigation['metrics']['OSR'],
            false_stop=navigation['metrics']['False_Stop_Rate'],checkpoint_sha256=metrics['checkpoint_sha256']))
        base='A3_relation000' if name.startswith('A3_relation') else 'A3_seed1' if name=='E1_seed1' else 'A3_seed0'
        summary['paired_static'][name]=paired(source/'static_seen'/(base+'_predictions.jsonl'),
            source/'static_seen'/(name+'_predictions.jsonl'),'R@1','map_name')
    for name,model in summary['models'].items():
        base='A3_relation000' if name.startswith('A3_relation') else 'A3_seed1' if name=='E1_seed1' else 'A3_seed0'
        summary['paired_closedloop'][name]=paired(ROOT/summary['models'][base]['raw_closedloop_predictions'],
            ROOT/model['raw_closedloop_predictions'],'metrics.SR','map_name')
    (destination/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (destination/'ablation.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for name in ('E1_seed0','E1_seed1','A3_relation000','A3_relation001','A3_relation005','A4_full','A4_self','A4_zero_geometry','A4_reliable_relation'):
        history=json.loads((destination/(name+'_training.json')).read_text())
        epochs=[row['epoch'] for row in history]
        for ax,(split,key) in zip(axes.flat,(('train','loss'),('seen','loss'),('seen','R@1/20m'),('seen','NE_m'))):
            ax.plot(epochs,[row[split][key] for row in history],label=name)
            ax.set_title(split+' '+key);ax.set_xlabel('Epoch');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(destination/'training_curves.png',dpi=160);plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
