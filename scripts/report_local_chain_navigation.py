"""Summarize measured paired TRAIN closed-loop outcomes, without Val claims."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    methods=('original','a3','real','zero','shuffle')
    data={m:[json.loads(x) for x in (a.run/m/'predictions.jsonl').read_text().splitlines()] for m in methods}
    metrics={m:json.loads((a.run/m/'metrics.json').read_text()) for m in methods}
    indexed={m:{r['index']:r for r in rows} for m,rows in data.items()}
    assert all(set(v)==set(indexed['original']) for v in indexed.values())
    for i in indexed['original']:
        assert all(np.array_equal(indexed[m][i]['poses'][0],indexed['original'][i]['poses'][0]) for m in methods)
    latency={}
    for method,rows in data.items():
        times=[t*1000 for r in rows for t in r['decision_seconds'][1:]]
        jumps=np.concatenate([np.linalg.norm(np.diff(np.array([d['selected_goal_xy'] for d in r['step_diagnostics']]),axis=0),axis=-1) for r in rows if len(r['step_diagnostics'])>1])
        latency[method]=dict(noninitial_decision_mean_ms=float(np.mean(times)),noninitial_decision_median_ms=float(np.median(times)),
            mean_policy_total_seconds=float(np.mean([sum(r['decision_seconds']) for r in rows])),
            median_goal_jump_m=float(np.median(jumps)),
            last_predicted_goal_distance_median_m=float(np.median([r['step_diagnostics'][-1]['predicted_goal_distance'] for r in rows])))
    diff=np.array([indexed['real'][i]['metrics']['SR']-indexed['original'][i]['metrics']['SR'] for i in indexed['original']])
    rng=np.random.default_rng(0);ci=np.quantile([rng.choice(diff,len(diff),replace=True).mean() for _ in range(5000)],[.025,.975])
    atomic_json(dict(methods=metrics,latency_and_stability=latency,paired_real_vs_original_sr_gain=float(diff.mean()),
        target_bootstrap_ci95=ci.tolist(),resamples=5000,seed=0,
        real_shuffle_sr_disagreements=sum(indexed['real'][i]['metrics']['SR']!=indexed['shuffle'][i]['metrics']['SR'] for i in indexed['real']),
        warning='24 TRAIN targets / 3 maps. Not official Val/Test; CI excludes map/model-selection uncertainty; no significant improvement claim.'),a.output/'summary.json')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,label,scale in [(axes[0],'SR','TRAIN closed-loop SR (%)',100),(axes[1],'SPL','SPL',1),(axes[2],'steps','Mean navigation steps',1)]:
        values=[metrics[m]['metrics'][key]*scale for m in methods]
        ax.bar(methods,values,color=['#2277aa','#888888','#cc4444','#aaaaaa','#ee9933']);ax.set_ylabel(label);ax.tick_params(axis='x',labelrotation=25)
    fig.suptitle('Frozen paired TRAIN diagnostic: same 24 targets; no Val/Test');fig.tight_layout();fig.savefig(a.output/'navigation_comparison.png',dpi=160);plt.close(fig)
    # Stop guard inputs are public diagnostics, no GT intervention.
    selected=[14344,19505,14401]
    fig,axes=plt.subplots(1,len(selected),figsize=(13,4))
    for ax,index in zip(axes,selected):
        for method in ('original','real','zero','shuffle'):
            values=[max(d['predicted_goal_distance'],1e-6) for d in indexed[method][index]['step_diagnostics']]
            ax.plot(range(len(values)),values,label=method)
        ax.axhline(.001,color='black',ls='--',label='Original arrival guard');ax.set_yscale('log')
        ax.set(xlabel='Actual decision step',ylabel='Distance to predicted Goal (m)',title=f'TRAIN {index}')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(a.output/'arrival_guard_diagnostics.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
