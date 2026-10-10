"""Scientific figures and all fixed Seen weak-semantic cases, real RGB only."""
import argparse
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from .evaluate_features import load
from experiments.bidirectional_core.cache import DEFAULT_DATA,jsonl
from graphnav.renderer import CityNavObservedViewRenderer


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True);p.add_argument('--episodes',required=True);p.add_argument('--data-root',default=DEFAULT_DATA);args=p.parse_args();runs=Path(args.runs);output=Path(args.output)
    if output.exists():raise ValueError('refusing figure overwrite')
    output.mkdir(parents=True);cv2.setNumThreads(1);rows=load(runs/'features_city_seen_judged');episodes={r['episode_id']:r for r in jsonl(args.episodes)};renderer=CityNavObservedViewRenderer(Path(args.data_root)/'rgbd',output/'renderer_private',image_size=384)
    preds={r['sample_id']:r for r in jsonl(runs/'semantic_seen_frozen/aerial_city_adapt_real_predictions.jsonl')};cases=[]
    for i,row in enumerate(rows):
        m=row['metadata'];regions=m['roi_objects'];scores=preds[m['sample_id']]['scores'];judged=row['labels']['semantic_valid_mask'];positive=row['labels']['positive_roi_mask'];rank=sorted((j for j,v in enumerate(judged) if v),key=lambda j:(-scores[j],regions[j]['object_id']))
        target=episodes[m['episode_id']]['private_labels']['target_object_id'];targets=[j for j,r in enumerate(regions) if r['object_id']==str(target)];true=targets[0] if targets else int(positive.nonzero()[0]);top=rank[0]
        observation=renderer.current_rgbd(m['map_name'],np.asarray(m['actual_camera_pose'],np.float64),float(row['public']['ground_level']));rgb=cv2.cvtColor(observation['bgr'],cv2.COLOR_BGR2RGB)
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        axes[0].imshow(rgb)
        for j,color in [(true,'lime'),(top,'red')]:
            x,y,c,d=regions[j]['bbox'];axes[0].add_patch(Rectangle((x,y),c-x,d-y,fill=False,color=color,linewidth=2))
        axes[0].set_title('Current observed RGB (384 source)')
        for ax,j,title in [(axes[1],true,'Associated original target (weak label)'),(axes[2],top,'VEN selected judged region')]:
            x,y,c,d=regions[j]['bbox'];ax.imshow(rgb[y:d,x:c]);ax.set_title(title+f'\n{regions[j]["object_type"]}, native minside {regions[j]["native_min_side"]}px')
        for ax in axes:ax.set_axis_off()
        fig.suptitle(row['phrase']+' | '+('weak-label correct' if positive[top] else 'weak-label failure'),fontsize=11);fig.tight_layout();name=f'case_{i:02}.png';fig.savefig(output/name,dpi=110);plt.close(fig)
        cases.append(dict(file=name,sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],cutoff_step=m['cutoff_step'],instruction=m['instruction'],phrase=row['phrase'],annotation_status='weak language/geometry association, not human attribute ground truth',original_target_object_id=target,predicted_object_id=regions[top]['object_id'],weak_correct=bool(positive[top]),roi_bbox=regions[true]['bbox'],native_min_side=regions[true]['native_min_side']))
    (output/'cases.json').write_text(json.dumps(cases,indent=2))
    sem=json.loads((runs/'semantic_seen_frozen/metrics.json').read_text());foundation=json.loads((runs/'foundation_selection_frozen.json').read_text());models=[]
    for name,label in [('native_real','SigLIP2 native'),('city_only_projection_real','City head'),('aerial_projection_real','Aerial head'),('aerial_city_adapt_real','Aerial + City'),('aerial_city_token_adapt_real','Token attention')]:models.append((label,sem['models'][name]['forward_r1']))
    for name,label in [('georsclip','GeoRSCLIP'),('skyclip','SkyCLIP')]:
        d=json.loads((runs/'stage1'/f'{name}_seen_frozen'/'metrics.json').read_text());models.append((label,d['models'][foundation['models'][name]['method']]['forward_r1']))
    fig,ax=plt.subplots(figsize=(10,4));means=np.array([s['mean'] for _,s in models])*100;intervals=np.array([s['map_cluster_95ci'] for _,s in models])*100
    ax.bar(range(len(models)),means,color='#386fa4');ax.errorbar(range(len(models)),means,yerr=np.stack([means-intervals[:,0],intervals[:,1]-means]),fmt='none',ecolor='black',capsize=3);ax.set_xticks(range(len(models)),[label for label,_ in models],rotation=20,ha='right');ax.set_ylabel('Weak language-region R@1 (%)');ax.set_ylim(0,105);ax.set_title('Fixed Seen: 14 states, 10 families, 9 maps; map-bootstrap CI');fig.tight_layout();fig.savefig(output/'semantic_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));groups=['real','zero','shuffled'];means=[sem['models']['aerial_city_adapt_'+g]['forward_r1']['mean']*100 for g in groups];ax.bar(groups,means,color=['#386fa4','#a4a4a4','#d17c3f']);ax.set_ylim(0,100);ax.set_ylabel('Weak language-region R@1 (%)');ax.set_title('RGB interventions retain destination geometry');fig.tight_layout();fig.savefig(output/'rgb_controls.png',dpi=150);plt.close(fig)
    runtime=json.loads((runs/'runtime_verified/metrics.json').read_text())['models'];fig,axes=plt.subplots(1,2,figsize=(9,4));names=['siglip2','georsclip','skyclip'];axes[0].bar(names,[runtime[n]['single_roi']['mean_ms'] for n in names]);axes[0].set_ylabel('Actual single ROI latency (ms)');axes[1].bar(names,[runtime[n]['peak_gpu_bytes']/1024**3 for n in names]);axes[1].set_ylabel('Peak allocated GPU (GiB)');fig.tight_layout();fig.savefig(output/'runtime_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));xs=['lt32','32to63','64plus'];width=.35
    for offset,name,label in [(-width/2,'native_real','Native'),(width/2,'aerial_city_adapt_real','Aerial + City')]:
        g=sem['groups'][name]['pixel_group'];ax.bar(np.arange(3)+offset,[g[x]['forward_r1']['mean']*100 for x in xs],width,label=label)
    ax.set_xticks(range(3),['<32 px (n=6)','32–63 px (n=4)','≥64 px (n=4)']);ax.set_ylim(0,100);ax.set_ylabel('Weak R@1 (%)');ax.legend();ax.set_title('Original-target native size; tiny, confounded groups');fig.tight_layout();fig.savefig(output/'native_pixel_groups.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(real_cases=len(cases),figures=4,output=str(output))))


if __name__=='__main__':main()
