"""Posthoc paired failure trajectories, using TRAIN labels only for plotting."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve existing plots')
    variants={'Original':('navigation_dev25_retarget_coarse','original'),
              'FrozenV3':('navigation_dev25_envelope','real'),
              'SoftInstance':('navigation_dev25_retarget_soft_instance','real'),
              'SoftShuffle':('navigation_dev25_retarget_soft_instance','rgb-shuffle')}
    cases={name:{r['index']:r for r in map(json.loads,(a.root/folder/method/'predictions.jsonl').read_text().splitlines())} for name,(folder,method) in variants.items()}
    targets={r['frame_key'].split('_')[1]:np.asarray(r['target_world'][:2]) for r in map(json.loads,(a.root/'fixed_yaw_train400_s1/labels_dev.jsonl').read_text().splitlines())}
    failures=[i for i in cases['SoftInstance'] if cases['FrozenV3'][i]['metrics']['SR'] and not cases['SoftInstance'][i]['metrics']['SR']]
    a.output.mkdir(parents=True);summary={}
    for index in failures:
        target=targets[f'{index:05d}'];fig,axes=plt.subplots(1,2,figsize=(12,5));summary[index]={}
        for name,rows in cases.items():
            row=rows[index];poses=np.asarray(row['poses']);goals=np.asarray([d['selected_goal_xy'] for d in row['step_diagnostics']]);errors=np.linalg.norm(goals-target,axis=-1)
            axes[0].plot(poses[:,0],poses[:,1],'.-',label=name)
            axes[1].plot(errors,'.-',label=name)
            summary[index][name]=dict(metrics=row['metrics'],goal_errors_m=errors.tolist(),posthoc=row['posthoc_goal_analysis'])
        axes[0].add_patch(plt.Circle(target,20,fill=False,color='black',linestyle='--'));axes[0].scatter(*target,marker='*',s=100,c='black');axes[0].set_aspect('equal');axes[0].set_xlabel('World X (m)');axes[0].set_ylabel('World Y (m)');axes[0].legend()
        axes[1].axhline(20,color='black',linestyle='--');axes[1].set_xlabel('Causal observation step');axes[1].set_ylabel('Predicted Goal error (m)');axes[1].legend()
        fig.suptitle(f'TRAIN {index}: paired closed-loop failure, GT posthoc only');fig.tight_layout();fig.savefig(a.output/f'failure_{index}.png',dpi=160);plt.close(fig)
    (a.output/'failure_details.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
