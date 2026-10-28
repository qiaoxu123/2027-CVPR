"""Export verified complete revision curves; never runs model inference."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    base=Path('runs/a2_waldo_landmark_20261027/revision_v1/training');out=Path('reports/a2_execution_20261027')
    h=json.loads((base/'history.json').read_text());verified=json.loads((base/'artifact_verification.json').read_text())
    if len(h)!=25 or verified['status']!='revision_all25_artifacts_verified':raise ValueError('Verified full25 training required')
    keys=('loss','known_candidate_mrr','full_pool_verified_target_top1','proposal_r1','map_r1','a0_r1','map_ne','a0_ne','corrected','broken','gate_mean','views','optimizer_updates','seconds')
    columns=['epoch','stage']+[f'{s}_{k}' for s in ('train','dev') for k in keys]
    with (out/'revision_v1_training_curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in h:writer.writerow(dict(epoch=r['epoch'],stage=r['stage'],**{f'{s}_{k}':r[s][k] for s in ('train','dev') for k in keys}))
    fig,axes=plt.subplots(2,3,figsize=(14,8));x=[r['epoch'] for r in h]
    for s in ('train','dev'):axes[0,0].plot(x,[r[s]['loss'] for r in h],label=s)
    axes[0,0].set_title('Total loss (objectives change at epoch21)')
    for key in ('known_candidate_mrr','full_pool_verified_target_top1'):axes[0,1].plot(x,[r['dev'][key] for r in h],label=key)
    axes[0,1].set_title('Different ranking populations; not navigation SR');axes[0,1].set_ylim(0,1)
    for key in ('map_r1','a0_r1'):axes[0,2].plot(x,[100*r['dev'][key] for r in h],label=key)
    axes[0,2].set_title('Dev cached-grid R@1 within20m (%)')
    for key in ('map_ne','a0_ne'):axes[1,0].plot(x,[r['dev'][key] for r in h],label=key)
    axes[1,0].set_title('Dev cached-grid argmax NE (m)')
    for key in ('corrected','broken'):axes[1,1].plot(x,[r['dev'][key] for r in h],label=key)
    axes[1,1].set_title('Dev corrected/broken observations, uncalibrated')
    axes[1,2].plot(x,[r['dev']['gate_mean'] for r in h],label='mean gate');axes[1,2].set_title('Benefit gate prior to independent calibration')
    for ax in axes.flat:
        ax.axvspan(20.5,25.5,color='gray',alpha=.12);ax.axvline(verified['selected_epoch'],color='black',linestyle=':',linewidth=1)
        ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle(f'A2 revision_v1 seed0: all20+5 complete; selected epoch{verified["selected_epoch"]}; TRAIN-dev only')
    fig.tight_layout();fig.savefig(out/'revision_v1_training_curves.png',dpi=170);fig.savefig(out/'revision_v1_training_curves.pdf');plt.close(fig)
    (out/'REVISION_TRAINING_COMPLETE.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2))
    print('Exported all25 verified training curves')


if __name__=='__main__':main()
