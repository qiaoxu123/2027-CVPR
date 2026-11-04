"""Plot actual post-rollout decisions; thresholds are shown, never changed."""
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[json.loads(x) for x in (a.run/'predictions.jsonl').read_text().splitlines()]
    raw=a.output/'native_rgb';raw.mkdir(exist_ok=True)
    for index in (0,1,13,34,39,43):
        group=[r for r in rows if r['index']==index]
        fig,ax=plt.subplots(4,1,figsize=(10,10),sharex=True,constrained_layout=True)
        for r in group:
            ds=r['step_diagnostics'];steps=np.arange(len(ds))
            ax[0].plot(steps,[d['public_landmark_goal_distance_m'] for d in ds],'.-',label=r['method'])
            ax[1].plot(steps,[d['predicted_goal_distance'] for d in ds],'.-',label=r['method'])
            ax[2].plot(steps,[d['stop_probability'] for d in ds],'.-',label=r['method'])
            ax[3].plot(steps,[d['landmark_visual_gain'] for d in ds],'.-',label=r['method'])
            if r['method']=='TrackedPGVB':
                for step,d in enumerate(ds):
                    if step==0 or d['landmark_tracking_state']!=ds[step-1]['landmark_tracking_state']:
                        ax[3].annotate(d['landmark_tracking_state'],(step,d['landmark_visual_gain']),fontsize=7,rotation=35)
                for step in sorted({0,len(ds)-1,*([r['tracking_analysis']['first_tracking_step']] if r['tracking_analysis']['first_tracking_step'] is not None else [])}):
                    shutil.copy2(a.run/'decisions'/f'{index:02d}_TrackedPGVB'/f'step_{step:02d}.png',raw/f'task_{index:02d}_step_{step:02d}.png')
        ax[0].axhline(20,color='black',ls='--');ax[0].set_ylabel('Goal / Polygon distance (m)')
        ax[1].axhline(.001,color='black',ls='--');ax[1].set_ylabel('UAV / selected Goal (m)')
        ax[2].axhline(.3,color='black',ls='--');ax[2].set_ylabel('Learned Stop probability')
        ax[3].set_ylabel('Visual gain');ax[3].set_xlabel('Decision step')
        for panel in ax:panel.legend(fontsize=7);panel.grid(alpha=.2)
        fig.suptitle(f'{index}: {group[0]["landmark_name"]}')
        fig.savefig(a.output/f'transitions_{index:02d}.png',dpi=130);plt.close(fig)

if __name__=='__main__':main()
