"""Render saved results and local public graphs; GT displayed offline only."""
import json,textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from .analyze import read
from .evaluate import OUT
from experiments.language_object_graph.run import load
from experiments.language_object_graph.objects import ObjectGraph
ASSETS=Path('docs/assets/role_aware_object_graph')
def main():
 ASSETS.mkdir(parents=True,exist_ok=True);a=json.loads((OUT/'analysis.json').read_text());methods=['A1','A2','A3','A4'];fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,key,title in zip(axes,['object_r1','goal_r1'],['Object identity R@1','Goal R@1 (20m)']):
  vals=[a['seen']['methods'][m]['metrics'][key]['mean']*100 for m in methods];ax.bar(methods,vals,color=['#888888','#4392c4','#20a47f','#c58b23']);ax.set(ylabel='%',title=title);ax.set_ylim(0,60)
  for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.2f}',ha='center')
 fig.suptitle('P5.1 fixed Seen confirmation: 128 independent families');fig.tight_layout();fig.savefig(ASSETS/'seen_comparison.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,3.5));vals=[];errors=[]
 for m in methods[1:]:
  d=a['seen']['methods'][m]['paired_vs_P5']['goal_r1'];v=d['mean']*100;lo,hi=np.array(d['map_cluster_95ci'])*100;vals.append(v);errors.append([v-lo,hi-v])
 ax.errorbar(vals,range(3),xerr=np.array(errors).T,fmt='o',capsize=4);ax.axvline(0,color='gray');ax.axvline(1,color='green',ls='--',label='Required +1 pp');ax.set(yticks=range(3),yticklabels=methods[1:],xlabel='Goal R@1 difference vs P5 A2 (pp)',title='Paired map bootstrap 95% CI, 2,000 replicates');ax.legend();fig.tight_layout();fig.savefig(ASSETS/'paired_goal_ci.png',dpi=170);plt.close(fig)
 old=read(OUT/'dev_A1.jsonl.gz');new=read(OUT/'dev_A4_rule_w0.5.jsonl.gz');states,_,_=load('dev');stateby={s['metadata']['sample_id']:s for s in states};cases=[]
 # Save one text scope case plus actual recovery/damage cases. No parameter selection.
 indexes=[0]
 for condition in (lambda o,n:not o['metrics']['goal_r1'] and n['metrics']['goal_r1'],lambda o,n:o['metrics']['goal_r1'] and not n['metrics']['goal_r1']):
  j=next((i for i,(o,n) in enumerate(zip(old,new)) if condition(o,n)),None)
  if j is not None and j not in indexes:indexes.append(j)
 for number,j in enumerate(indexes):
  o,n=old[j],new[j];s=stateby[n['sample_id']];g=ObjectGraph.restore(torch.load(Path('runs/language_object_graph/graphs')/(n['map_name']+'.pt'),weights_only=True));fig,axes=plt.subplots(1,3,figsize=(14,5));axes[0].axis('off');l=n['language_graph'];lines=['Raw instruction:','\n'.join(textwrap.wrap(n['instruction'],42)),'','P5 target:',str(o['language_graph']['entities'][0]['phrase']),'P5 target names:',str(o['language_graph']['entities'][0]['named_names']),'','Role-aware target:',l['entities'][0]['phrase'] if l['entities'] else 'Unknown','Scoped factors:']
  for f in l['factors']:lines.append(f"{l['entities'][f['subject']]['head_text']} --{f['kind']}--> "+'/'.join(l['entities'][i]['head_text'] for i in f['anchors']))
  axes[0].text(0,1,'\n'.join(lines),va='top',fontsize=8,wrap=True)
  p=s['public'];b=np.array(p['bounds']);xy=b[:2]+np.array(p['grid_xy'])*(b[2:]-b[:2]);gt=np.array(n['offline_label']['target_world']);refids=set()
  for e in l['entities'][1:]:refids.update(e['named_ids'])
  for ax,r,title in zip(axes[1:],[o,n],['P5 A2','Role-aware A4']):
   values=np.array(r['goal_logits']);values=np.exp(values-values.max());values/=values.sum();plot=ax.scatter(xy[:,0],xy[:,1],c=values,s=13,cmap='viridis');top=g.byid[str(r['object_ids'][0])];center=g.centers[top];local=g.tree.query(__import__('shapely').geometry.Point(center).buffer(100));edges=g.edge_index;mask=np.isin(edges[0],local)&np.isin(edges[1],local)
   for u,v in edges[:,mask].T[::max(1,mask.sum()//150)]:ax.plot(g.centers[[u,v],0],g.centers[[u,v],1],color='silver',alpha=.2,lw=.4)
   for k in local:
    shape=g.shapes[k]
    if shape.geom_type=='Polygon':v=np.array(shape.exterior.coords);ax.plot(v[:,0],v[:,1],color='orange' if k==top else '#888',lw=1.5 if k==top else .3)
   for k in refids:ax.scatter(*g.centers[k],marker='s',s=50,color='cyan')
   ax.scatter(*gt,marker='*',s=130,color='red',label='GT offline only');ax.scatter(*xy[r['goal_order'][0]],marker='x',s=80,color='white',label='Goal Top1');ax.set(title=f"{title}: success={int(r['metrics']['goal_r1'])}, NE={r['metrics']['ne']:.1f}m",xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal');ax.legend(fontsize=7)
  fig.suptitle(n['sample_id'],fontsize=9);fig.tight_layout();name=f'case_{number}.png';fig.savefig(ASSETS/name,dpi=160);plt.close(fig);cases.append(dict(sample_id=n['sample_id'],instruction=n['instruction'],P5=o['language_graph'],role_aware=l,old_metrics=o['metrics'],new_metrics=n['metrics'],figure=name,label_status='Goal GT only offline visualization; no verified reference identity'))
 (ASSETS/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
 audit=read(OUT/'audit_A4_rule_w0.5.jsonl.gz');fail=[r for r in audit if not r['role_correct'] or any(not c['primary_scope_correct'] for c in r['cases'])];(ASSETS/'parse_failures.json').write_text(json.dumps(fail,ensure_ascii=False,indent=2)+'\n')
 print('Figures and offline cases saved:',ASSETS)
if __name__=='__main__':main()
