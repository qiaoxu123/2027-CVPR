"""Archive small actual evidence and draw standalone scientific result figures."""
import argparse,gzip,json,shutil,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .evaluate import load_prepared,write_json
from .coverage import stats
from experiments.bidirectional_core.stats import paired_difference


def run(runroot,output):
    runroot=Path(runroot);output=Path(output)
    if output.exists():raise ValueError('no overwrite')
    output.mkdir(parents=True);archive=output/'evidence';archive.mkdir();manifest={}
    for directory in ['evaluation_dev','evaluation_seen','coverage_dev','coverage_seen','association_audit_dev','association_audit_seen','coverage_oracle_dev','coverage_oracle_seen']:
        source=runroot/directory;dest=archive/directory;dest.mkdir()
        shutil.copy2(source/'metrics.json',dest/'metrics.json')
        # Deterministic gzip archives retain every prediction, avoiding large
        # text duplicates in Git. Full original files remain local in runs.
        for path in sorted(source.glob('*predictions.jsonl')):
            target=dest/(path.name+'.gz')
            with target.open('wb') as f:
                with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(path.read_bytes())
        manifest[directory]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()}
    for name in ['freeze.json','regression_tests.txt']:
        shutil.copy2(runroot/name,archive/name)
    shutil.copy2(runroot/'calibration_dev_verified/frozen_config.json',archive/'frozen_config.json')
    for split in ['dev','seen']:shutil.copy2(runroot/f'prepared_{split}/manifest.json',archive/f'prepared_{split}_manifest.json')
    p0=json.loads((runroot/'evaluation_seen/metrics.json').read_text());p1=json.loads((runroot/'coverage_seen/metrics.json').read_text())
    methods=['centroid','polygon_overlap','distance_soft','instance'];labels=['Centroid','Polygon overlap','Distance soft','Instance max']
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
    for i,kind in enumerate(['real','zero','shuffled']):
        vals=[100*p0['models'][m+'_selected_'+kind]['r1']['mean'] for m in methods];ax.bar(x+(i-1)*.23,vals,.23,label=kind)
    ax.axhline(100*p0['models']['geometry']['r1']['mean'],color='black',ls='--',label='Geometry')
    ax.set_xticks(x,labels);ax.set_ylabel('Goal R@1 (%)');ax.set_ylim(0,65);ax.legend(ncol=4);ax.set_title('Frozen Seen: 640 prefixes / 128 families');fig.tight_layout();fig.savefig(output/'goal_controls.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for i,history in enumerate(['current','history']):
        vals=[100*p0['models'][m+'_oracle_'+history+'_all_cached_rois']['r1']['mean'] for m in methods];ax.bar(x+(i-.5)*.33,vals,.33,label='Identity oracle '+history)
    ax.axhline(100*p0['models']['geometry']['r1']['mean'],color='black',ls='--',label='Geometry')
    ax.set_xticks(x,labels);ax.set_ylabel('Diagnostic Goal R@1 (%)');ax.legend();fig.tight_layout();fig.savefig(output/'oracle_association.png',dpi=160);plt.close(fig)
    policies=['current','cached_history','dense_causal_history','hold','approach','descend','lateral','belief_active'];names=['Current','Sparse history','Dense causal','Hold x2','Approach','Descend','Lateral','Belief active']
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(8)
    for i,key in enumerate(['target_valid','target_credible']):
        values=[100*p1['models'][s][key]['mean'] for s in policies];ax.bar(x+(i-.5)*.34,values,.34,label=key)
    ax.set_xticks(x,names,rotation=20);ax.set_ylabel('Observed target coverage (%)');ax.legend();ax.set_title('Geometry/depth quality proxy; semantics unverified');fig.tight_layout();fig.savefig(output/'coverage.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,m in enumerate(methods):
        s=p0['models'][m+'_selected_real'];axes[0].bar(i-.15,s['recovery']['count'],.3,color='tab:green');axes[0].bar(i+.15,s['damage']['count'],.3,color='tab:red')
    axes[0].set_xticks(np.arange(4),labels,rotation=20);axes[0].set_ylabel('Recovery (green) / Damage (red) states')
    costs=[p1['models'][s]['added_observation_seconds']['mean'] for s in policies];axes[1].bar(np.arange(8),costs);axes[1].set_xticks(np.arange(8),names,rotation=45);axes[1].set_ylabel('Extra rendered observation CPU seconds / prefix');fig.tight_layout();fig.savefig(output/'safety_cost.png',dpi=160);plt.close(fig)
    # Derive actual motion steps separately from equal observation-slot budget.
    data=load_prepared(runroot/'prepared_seen');states={s['sample_id']:s for s in data['public']};motion={}
    for policy in policies:
        rows=[]
        for line in (runroot/'coverage_seen'/(policy+'_predictions.jsonl')).read_text().splitlines():
            row=json.loads(line);pose=states[row['sample_id']]['pose'];count=0
            for v in row['visited_poses']:
                v=np.array(v);count+=int(np.linalg.norm(v[:3]-pose[:3])>1e-6);pose=v
            row['metrics']['actual_extra_motion_steps']=count;rows.append(row)
        from experiments.bidirectional_core.stats import summarize_rows
        motion[policy]=summarize_rows(rows,'actual_extra_motion_steps',seed=20261011,repetitions=2000)
    write_json(archive/'motion_steps.json',motion);write_json(archive/'archive_sha256.json',manifest)
    print(json.dumps(dict(files=len(list(output.rglob('*'))),bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.runs,a.output)
