"""Export complete additional20+5 head curves, never navigation success claims."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.a2_waldo_landmark_belief.paths import ROOT,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE


def main():
    source=BASE/'revision_v2/head_training';history=json.loads((source/'history.json').read_text())
    marker=json.loads((source/'training_complete.json').read_text());proof=json.loads((source/'artifact_verification.json').read_text())
    if [r['epoch'] for r in history]!=list(range(1,26)) or not proof['all_dev_epochs_recomputed'] or marker['checkpoint_sha256']!=sha(source/'best.pt'):
        raise ValueError('All25 verified head epochs required')
    output=ROOT/'reports/a2_execution_20261027';output.mkdir(parents=True,exist_ok=True)
    prefix=output/'direct_risk_v2_training_curves';rows=[]
    for r in history:
        t,d=r['train'],r['dev'];rows.append(dict(epoch=r['epoch'],stage=r['stage'],train_classification=t['classification'],dev_classification=d['classification'],
            train_risk_cost=t['risk_cost'],train_objective=t['loss'],dev_gain_harm_auc=d['gain_harm_auc'],dev_corrected=d['corrected'],dev_broken=d['broken'],
            dev_group_r1_difference=d['mean_r1_difference'],dev_supported_groups=d['supported_groups'],train_views=t['views'],dev_views=d['views'],optimizer_updates=t['optimizer_updates']))
    with prefix.with_suffix('.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    x=[r['epoch'] for r in rows];fig,axes=plt.subplots(2,3,figsize=(16,8))
    axes=axes.ravel()
    for key,label in (('train_classification','TRAIN batch-weighted CE'),('dev_classification','dev global weighted CE')):axes[0].plot(x,[r[key] for r in rows],label=label)
    axes[0].legend();axes[0].set_title('Classification loss (same head; no foundation training)')
    axes[1].plot(x,[r['train_risk_cost'] for r in rows]);axes[1].set_title('Risk cost (added to objective at epoch21)')
    axes[2].plot(x,[100*r['dev_group_r1_difference'] for r in rows]);axes[2].axhline(0,color='grey');axes[2].set_title('DEV chosen operating point R@1 difference (pp)')
    axes[3].plot(x,[r['dev_corrected'] for r in rows],label='corrected views');axes[3].plot(x,[r['dev_broken'] for r in rows],label='broken views');axes[3].legend();axes[3].set_title('DEV selected-point counts, NOT independent validation')
    axes[4].plot(x,[r['dev_gain_harm_auc'] for r in rows]);axes[4].set_title('DEV gain-vs-harm AUC (auxiliary criterion)')
    axes[5].plot(x,[r['dev_supported_groups'] for r in rows]);axes[5].axhline(30,color='grey',linestyle=':');axes[5].set_title('DEV chosen-point supported target groups')
    for ax in axes:
        ax.axvspan(20.5,25.5,color='grey',alpha=.12);ax.axvline(marker['selected_epoch'],color='black',linestyle=':',alpha=.6);ax.set_xlabel('Additional head epoch');ax.grid(alpha=.2)
    fig.suptitle(f"Direct risk head seed0: ALL20+5 complete, selectedepoch{marker['selected_epoch']}; previous relation20+5 frozen; NOT navigation SR")
    fig.tight_layout();fig.savefig(prefix.with_suffix('.png'),dpi=150);fig.savefig(prefix.with_suffix('.pdf'));plt.close(fig)
    save(dict(proof,additional_schedule='20risk discrimination+5belief risk refinement; prior relation20+5 preserved',
              history_sha256=sha(source/'history.json'),protocol_sha256=sha(source/'protocol.json')),output/'DIRECT_RISK_V2_TRAINING_COMPLETE.json')
    save(json.loads((BASE/'revision_v2/cuda_smoke.json').read_text()),output/'DIRECT_RISK_V2_CUDA_SMOKE.json')
    print('Exported all25 additionalhead curves and actual verification report',flush=True)


if __name__=='__main__':main()
