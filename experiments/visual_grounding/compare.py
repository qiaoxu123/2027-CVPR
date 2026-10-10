"""Aggregate real predictions and paired model differences; never fill missing metrics."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from experiments.visual_grounding.freeze_dataset import read,sha
from experiments.visual_grounding.statistics import intervals


def event(r):
    return tuple(str(r.get(k,'')) for k in ['task','split','category','map_name','object_id','sample_id','query_id','from_label','to_label'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/visual_grounding_benchmark');p.add_argument('--models',nargs='+',default=['remoteclip_vitl14','remoteclip_vitb32','regionclip_rn50','siglip2_large384']);a=p.parse_args();root=Path(a.root);rows=[];models={};predictions={};manifest={}
    for name in a.models:
        folder=root/name;metrics={}
        for file in folder.glob('*/metrics_*.json'):
            if file.name not in ['metrics_development.json','metrics_unseen.json']:continue
            m=json.loads(file.read_text());metrics.update({s:v for s,v in m['results'].items() if v});metrics['telemetry']=m['telemetry'];metrics['frozen_sha256']=m['frozen_sha256']
        models[name]=metrics
        provenance=folder/'seen/model_provenance_benchmark.json';manifest[name]=json.loads(provenance.read_text()) if provenance.exists() else {'status':'not_complete'}
        preds=[]
        for file in folder.glob('*/predictions_*.jsonl'):
            if file.name in ['predictions_development.jsonl','predictions_unseen.jsonl']:preds.extend(read(file))
        predictions[name]={event(r):r for r in preds}
        for split in ['train_seen','val_seen','val_unseen']:
            for task,value in metrics.get(split,{}).items():
                rows.append(dict(model=name,split=split,task=task,n=value['n'],mean=value['mean'],macro_f1=value.get('macro_f1'),maps=value['maps'],independent_objects_or_queries=value.get('independent_objects_or_queries'),iid_low=value['iid_95ci'][0],iid_high=value['iid_95ci'][1],map_low=value['map_cluster_95ci'][0],map_high=value['map_cluster_95ci'][1],candidate_size_mean=value.get('candidate_size_mean')))
    baseref='siglip2_large384';paired={}
    for name,lookup in predictions.items():
        if name==baseref:continue
        common=set(lookup)&set(predictions.get(baseref,{}));grouped=defaultdict(list)
        for key in sorted(common):
            r=lookup[key];base=predictions[baseref][key];grouped[r['split'],r['task']].append((float(r['correct'])-float(base['correct']),r['map_name']))
        paired[name+'_minus_'+baseref]={str(k):intervals([v[0] for v in vs],[v[1] for v in vs]) for k,vs in grouped.items()}
    out=root/'comparisons';out.mkdir(exist_ok=True)
    with (root/'comparison.csv').open('w') as f:
        if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    auxiliary={}
    for key,pattern in [('proposal','regionclip_rn50/proposals_*/proposal_metrics_*.json'),('proposal_matched','regionclip_rn50/proposals_*/matched_comparison.json'),('patch_identity','siglip2_large384/*/patch_retrieval_*.json'),('a3','a3_reranking/summary.json'),('finetune','finetune_pilot/gate.json'),('runtime','*/runtime_benchmark.json'),('precision','*/precision_check.json')]:
        auxiliary[key]={str(file.relative_to(root)):json.loads(file.read_text()) for file in root.glob(pattern)}
    for name in a.models:
        for label,filename in [('precision_check','precision_check.json'),('runtime_measurement','runtime_benchmark.json'),('environment','environment.json')]:
            file=root/name/filename
            if file.exists():manifest[name][label]=json.loads(file.read_text())
    proposal_files=auxiliary['proposal']
    if proposal_files:
        manifest['regionclip_rn50']['proposal_status']='official RPN evaluated; geometric coverage separate from conditional semantic matching'
        manifest['regionclip_rn50']['proposal_evaluations']=list(proposal_files)
    (root/'metrics.json').write_text(json.dumps(dict(models=models,paired_differences=paired,auxiliary=auxiliary,verified_semantic_metrics=None,limitation='CityRefer language weak labels are exploratory. Independent image-verified labels unavailable; these are not proven fine-grained semantic accuracy.'),indent=2))
    (root/'model_manifest.json').write_text(json.dumps(manifest,indent=2))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    colors=['#187bcd','#62a6d6','#d98c27','#54a16c']
    for split in ['val_seen','val_unseen']:
        tasks=['weak_attribute','weak_hard_negative_attribute_phrase_r1','weak_hard_negative_instruction_r1','weak_counterfactual','cross_view_instance_r1']
        fig,axs=plt.subplots(1,len(tasks),figsize=(18,4))
        for ax,task in zip(axs,tasks):
            vals=[models.get(n,{}).get(split,{}).get(task,{}).get('mean',np.nan) for n in a.models]
            ax.bar(np.arange(len(a.models)),np.asarray(vals,dtype=float)*100,color=colors[:len(a.models)]);ax.set_xticks(np.arange(len(a.models)),['Remote L','Remote B','Region RN50','SigLIP2'],rotation=50,ha='right');ax.set_ylim(0,100);ax.set_title(task.replace('weak_','Weak ').replace('_',' '),fontsize=9);ax.set_ylabel('%');ax.grid(axis='y',alpha=.2)
        fig.suptitle(split+' — language weak labels; identity retrieval separate');fig.tight_layout();fig.savefig(out/f'{split}_accuracy.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4));kinds=['roof_color','building_color','vehicle_color','building_shape'];x=np.arange(len(kinds));width=.18
        for j,n in enumerate(a.models):
            values=[models.get(n,{}).get(split,{}).get('weak_attribute',{}).get('by_attribute_or_type',{}).get(k,{}).get('mean',np.nan) for k in kinds];ax.bar(x+(j-1.5)*width,np.asarray(values,dtype=float)*100,width,label=n,color=colors[j])
        ax.set_xticks(x,kinds);ax.set_ylim(0,100);ax.set_ylabel('Weak attribute accuracy %');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/f'{split}_attributes.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for j,(metric,label) in enumerate([('throughput_roi_s','Measured throughput ROI/s'),('peak_allocated_gib','Peak allocated GiB')]):
        vals=[models.get(n,{}).get('telemetry',{}).get(metric,np.nan) for n in a.models];axs[j].bar(np.arange(len(a.models)),vals,color=colors);axs[j].set_xticks(np.arange(len(a.models)),['Remote L','Remote B','Region RN50','SigLIP2'],rotation=30);axs[j].set_ylabel(label)
    fig.suptitle('Actual unified benchmark; source-frame regions vs ROI encoders');fig.tight_layout();fig.savefig(out/'runtime_memory.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(models=len(models),metric_rows=len(rows),paired_comparisons=len(paired))))
if __name__=='__main__':main()
