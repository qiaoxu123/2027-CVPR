"""Export complete A1 development curves and verify TensorBoard scalar records."""
import argparse
import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from graphnav.checkpointing import atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];audit={};history={};offset=0
    for stage,epochs in (('A',3),('B',3),('C',5)):
        folder=root/'A1_joint_rgb_geometry/stages'/stage
        records=json.loads((folder/'history.json').read_text());assert len(records)==epochs
        accumulator=EventAccumulator(str(folder/'tensorboard'));accumulator.Reload()
        tags={}
        for tag in ('train/total','dev/total','seen128/SR'):
            events=accumulator.Scalars(tag)
            assert len(events)==epochs and [e.step for e in events]==list(range(1,epochs+1))
            split,key=tag.split('/')
            assert all(math.isclose(event.value,record[split][key],rel_tol=1e-6,abs_tol=1e-6)for event,record in zip(events,records))
            tags[tag]=dict(points=len(events),steps=[e.step for e in events])
        history[stage]=records
        audit[stage]=dict(history_sha256=file_sha256(folder/'history.json'),scalars=tags)
        for record in records:
            assert record['train']['samples']==83140 and record['dev']['samples']==4372
            rows.append(dict(stage=stage,epoch=record['epoch'],global_epoch=offset+record['epoch'],train_total=record['train']['total'],dev_total=record['dev']['total'],seen128_SR=record['seen128']['SR'],dev_recall_at1_20m=record['dev']['belief_R@1/20m'],dev_NE_m=record['dev']['belief_NE_m']))
        offset+=epochs
    with (out/'training_curves.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        for ax,(stage,records)in zip(axes[0],history.items()):
            x=[r['epoch']for r in records]
            ax.plot(x,[r['train']['total']for r in records],'-o',color='#0072B2',label='Train')
            ax.plot(x,[r['dev']['total']for r in records],'--s',color='#D55E00',label='Internal dev')
            ax.set(title=f'Stage {stage}: {len(records)} full epochs',xlabel='Stage epoch',ylabel='Total loss',xticks=x);ax.legend();ax.grid(alpha=.2)
        for ax,(key,title,ylabel,multiplier)in zip(axes[1],[('seen128_SR','Seen128 development SR','SR (%)',100),('dev_recall_at1_20m','Internal-dev belief recall','R@1 within 20m (%)',100),('dev_NE_m','Internal-dev belief NE','NE (m)',1)]):
            ax.plot([r['global_epoch']for r in rows],[multiplier*r[key]for r in rows],'-o',color='#009E73',markersize=4)
            for boundary in (3.5,6.5):ax.axvline(boundary,color='.6',linestyle=':',linewidth=1)
            for center,label in ((2,'A'),(5,'B'),(9,'C')):ax.text(center,.97,label,transform=ax.get_xaxis_transform(),ha='center',va='top',color='.4')
            ax.set(title=title,xlabel='Cumulative epoch',ylabel=ylabel,xticks=range(1,12),xlim=(.6,11.4));ax.grid(alpha=.2)
        fig.suptitle('A1: Seed 0, complete A3/B3/C5',fontsize=15)
        fig.text(.5,-.025,'Development metrics only. Stage C adds trajectory and anchor losses; compare loss within each stage.',ha='center',fontsize=9)
        for suffix in ('png','pdf'):fig.savefig(out/('training_curves.'+suffix),dpi=180,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    atomic_json(dict(complete=True,epochs=dict(A=3,B=3,C=5),tensorboard_matches_history=True,stages=audit,files={name:file_sha256(out/name)for name in ('training_curves.csv','training_curves.png','training_curves.pdf')},scope='A1 full training development curves, not full-split validation conclusions'),out/'training_curve_verification.json')
    print(json.dumps(dict(complete=True,epochs=11,tensorboard_verified=True,output=str(out.resolve()))))

if __name__=='__main__':main()
