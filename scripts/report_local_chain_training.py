"""Plot measured offline TRAIN position metrics; never labels these as SR."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-directory',type=Path,required=True)
    a=p.parse_args();root=a.report_directory
    training=json.loads((root/'training.json').read_text())['history']
    confirm=json.loads((root/'confirmation.json').read_text())['results']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    epochs=[r['epoch'] for r in training]
    axes[0].plot(epochs,[r['loss'] for r in training],marker='o');axes[0].set(xlabel='Epoch',ylabel='TRAIN world-position loss',xticks=epochs)
    axes[1].plot(epochs,[r['dev']['goal_r1']*100 for r in training],marker='o',label='Local head')
    axes[1].axhline(training[0]['dev']['baseline_goal_r1']*100,color='gray',linestyle='--',label='Frozen A3')
    axes[1].set(xlabel='Epoch',ylabel='TRAIN-dev Goal R@1 (%) - NOT SR',xticks=epochs);axes[1].legend()
    fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=160);plt.close(fig)
    names=list(confirm);values=[confirm[n]['metrics']['goal_r1']*100 for n in names]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,vals,label,baseline in [(axes[0],values,'Goal R@1 (%) - NOT SR',confirm['real']['metrics']['baseline_goal_r1']*100),
                                   (axes[1],[confirm[n]['metrics']['goal_ne_m'] for n in names],'Goal NE (m)',confirm['real']['metrics']['baseline_goal_ne_m'])]:
        ax.bar(names,vals,color=['#2277aa','#999999','#dd8833','#8866aa','#44aa88']);ax.axhline(baseline,color='black',linestyle='--',label='Frozen A3')
        ax.set_ylabel(label);ax.tick_params(axis='x',labelrotation=25);ax.legend()
    fig.suptitle('Frozen confirmation: 24 TRAIN targets / 91 states; no closed-loop results')
    fig.tight_layout();fig.savefig(root/'confirmation_ablations.png',dpi=160);plt.close(fig)
    grouped=defaultdict(list)
    for row in confirm['real']['rows']:grouped[row['task_key']].append(row)
    delta=np.array([np.mean([float(r['success'])-float(r['baseline_success']) for r in rows]) for rows in grouped.values()])
    rng=np.random.default_rng(0)
    ci=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)],[.025,.975])
    (root/'uncertainty.json').write_text(json.dumps(dict(bootstrap_unit='physical target; equal target weighting',seed=0,resamples=5000,
        target_balanced_goal_r1_gain=float(delta.mean()),ci95=ci.tolist(),
        warning='24 targets / 3 maps. Does not cover map-generalization or selection uncertainty; not SR.'),indent=2)+'\n')


if __name__=='__main__':main()
