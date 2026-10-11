"""Scientific figures and actual observed Train RGB examples; no new metrics."""
import csv,gzip,json,shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .run import OUT,load
from .objects import ObjectGraph
from .pipeline import forward
from .reasoning import probability
from experiments.semantic_polygon_grounding.audit import save

ASSETS=Path('docs/assets/language_object_graph')
def main():
    ASSETS.mkdir(parents=True,exist_ok=True);metrics=json.loads((OUT/'metrics.json').read_text());audit=json.loads((OUT/'graph_audit.json').read_text())
    rows=list(csv.DictReader((OUT/'comparison.csv').open()));fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,split in enumerate(('dev','seen')):
        subset=[r for r in rows if r['split']==split];axes[j].bar([r['method'] for r in subset],[float(r['r1']) for r in subset],color=['#888','#5479b7','#338b61','#cd9a37'])
        axes[j].set(title=split+' Goal R@1 (independent families)',ylabel='R@1 (%)',ylim=(0,100));axes[j].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(ASSETS/'goal_comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3));names=['A1 vs A0','A2 vs A0','A2 vs A1'];split='seen';comp=metrics['comparisons'][split];analysis=json.loads((OUT/'analysis.json').read_text())
    vals=[comp['A1_w0.5']['goal_r1'],comp['A2_w0.5']['goal_r1'],analysis['seen']['relation_delta_vs_A1']['goal_r1']]
    means=np.array([100*v['mean'] for v in vals]);low=np.array([100*v['map_cluster_95ci'][0] for v in vals]);high=np.array([100*v['map_cluster_95ci'][1] for v in vals]);ax.errorbar(means,np.arange(3),xerr=[means-low,high-means],fmt='o',capsize=4,color='#338b61');ax.axvline(0,color='gray',ls='--');ax.set(yticks=np.arange(3),yticklabels=names,xlabel='Paired map-cluster 95% CI (percentage points)',title='Seen localization deltas: none exclude zero');fig.tight_layout();fig.savefig(ASSETS/'paired_ci.png',dpi=170);plt.close(fig)
    states,labels,_=load('train');state=next(s for s in states if 'between' in s['metadata']['instruction'].lower());m=state['metadata'];g=ObjectGraph.restore(torch.load(OUT/'graphs'/f"{m['map_name']}.pt",weights_only=True));result=forward(g,m['instruction'],state['public'],method='A2',weight=.5)
    nodes=g.local(result['world_grid'][np.argsort(result['static'])[-8:]],mandatory=result['mandatory'],limit=192);subset=set(map(int,nodes));segments=[]
    for a,b in g.edge_index.T:
        if a!=b and int(a) in subset and int(b) in subset:segments.append(g.centers[[a,b]])
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].add_collection(LineCollection(segments,alpha=.12,colors='gray',linewidths=.4))
    colors={'Building':'#338b61','Car':'#aa6d3b','TrafficRoad':'#4779a9','Ground':'#999'}
    for i in nodes:
        poly=g.shapes[i]
        if g.valid[i]:x,y=poly.exterior.xy;axes[0].plot(x,y,color=colors.get(g.types[i],'#555'),lw=.6)
    axes[0].scatter(g.centers[nodes,0],g.centers[nodes,1],s=4,c='#333');axes[0].autoscale();axes[0].set_aspect('equal');axes[0].set_title(f'Public local graph ({len(nodes)} nodes)')
    for ax,heat,title in zip(axes[1:],[probability(state['public']['static_logits']),probability(result['logits'])],['A0 cached belief','A2 object + continuous belief']):
        xy=result['world_grid'];im=ax.scatter(xy[:,0],xy[:,1],c=heat,s=18,cmap='magma');target=labels[m['sample_id']]['target_world'];ax.scatter(*target,c='cyan',marker='x',s=65,label='GT: offline only');ax.set_aspect('equal');ax.set_title(title);ax.legend(fontsize=7);fig.colorbar(im,ax=ax,fraction=.035)
    fig.suptitle('Train exemplar: same original 784 cells; GT only for illustration',fontsize=10);fig.tight_layout();fig.savefig(ASSETS/'graph_and_heatmap.png',dpi=180);plt.close(fig)
    save(ASSETS/'language_binding_example.json',dict(metadata=m,language_graph=result['language'],relation_hypotheses=result['factors'],node_index_to_public_id={str(i):g.ids[i] for i in result['mandatory']},offline_gt_not_model_input=labels[m['sample_id']]['target_world']))
    qa=json.loads((OUT/'rgb_qa'/'observations.json').read_text());fig,axes=plt.subplots(2,4,figsize=(12,6))
    for j,(ax,r) in enumerate(zip(axes.ravel(),qa['observations'])):
        import matplotlib.image as mpimg
        image=mpimg.imread(OUT/'rgb_qa'/f'{j:02d}_polygons.png');ax.imshow(image);ax.axis('off');ax.set_title(f"{r['metadata']['map_name']}\n{r['observed_objects']} accepted polygons",fontsize=8)
    fig.suptitle('Actual causal Train RGB + projected polygons (weak masks, not semantic truth)');fig.tight_layout();fig.savefig(ASSETS/'actual_rgb_gallery.png',dpi=160);plt.close(fig)
    # Preserve a native-resolution example showing vehicle-mask contamination.
    shutil.copyfile(OUT/'rgb_qa'/ '02_polygons.png',ASSETS/'vehicle_polygon_roi.png')
    examples=[]
    with gzip.open(OUT/'seen_A2_w0.5.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['metrics']['recovery'] or r['metrics']['damage']:examples.append({k:r[k] for k in ['sample_id','family_id','map_name','instruction','metrics','goal_target_rank','language_graph','relation_hypotheses']})
    save(ASSETS/'recovery_damage_cases.json',examples)
    fig,ax=plt.subplots(figsize=(10,3));names=list(audit);ax.bar(np.arange(len(names)),[audit[n]['nodes'] for n in names],color='#5479b7');ax.set(xticks=np.arange(len(names)),xticklabels=names,ylabel='Object nodes',title='All 34 public map object graphs');ax.tick_params(axis='x',labelrotation=90,labelsize=6);fig.tight_layout();fig.savefig(ASSETS/'graph_scale.png',dpi=180);plt.close(fig)
    shutil.copyfile(OUT/'comparison.csv',ASSETS/'comparison.csv')

if __name__=='__main__':main()
