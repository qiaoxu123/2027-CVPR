"""Post-hoc TRAIN scoring and source-map plots of sealed direct-relation predictions."""
import argparse,csv,html,json
from collections import Counter
from pathlib import Path
import cv2,numpy as np,rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import file_sha256
from graphnav.direction_belief_data import UNIT
from scripts.score_direction_belief import score
from scripts.audit_direction_frames import interval_status

METHODS=['anchor','nodirection_real','direction_marginal','direction_synthetic','direction_real']
def paired(base,cur):
 b=np.array(base['per_sample_NE'])<=20;c=np.array(cur['per_sample_NE'])<=20
 return dict(corrected=int((~b&c).sum()),damaged=int((b&~c).sum()),net=int(c.sum()-b.sum()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();o=a.cohort;seal=json.loads((o/'prediction_seal.json').read_text());assert seal['target_labels_read'] is False;assert file_sha256(o/'probabilities.npz')==seal['prediction_sha256'];assert file_sha256(o/'public.json')==seal['cohort_sha256'];assert file_sha256(o/'selection.json')==seal['selection_sha256'];rows=json.loads((o/'public.json').read_text());labels={r['index']:r for r in json.loads(a.labels.read_text())};pred=np.load(o/'probabilities.npz');span=np.array([r['span'] for r in rows]);bounds=np.array([r['bounds'] for r in rows]);targets=np.array([labels[r['index']]['target_world'][:2] for r in rows]);local=targets-bounds[:,:2];ids=np.array([r['direction_id'] for r in rows]);metrics={};details=[]
 for sub in ['fit','dev']:
  mask=np.array([r['subset']==sub for r in rows]);pack={}
  for m in METHODS:pack[m]=dict(all=score(pred[m][mask],span[mask],local[mask],pred['anchor'][mask],ids[mask]),by_word={w:score(pred[m][mask&(ids==k)],span[mask&(ids==k)],local[mask&(ids==k)],pred['anchor'][mask&(ids==k)],ids[mask&(ids==k)]) for k,w in enumerate(['north','south','east','west'])})
  pack['paired_vs_nodirection']=paired(pack['nodirection_real']['all'],pack['direction_real']['all']);pack['paired_vs_marginal']=paired(pack['direction_marginal']['all'],pack['direction_real']['all']);metrics[sub]=pack
 for j,r in enumerate(rows):
  b=bounds[j];yy,xx=np.meshgrid((np.arange(28)+.5)/28,(np.arange(28)+.5)/28,indexing='ij');grid=np.stack([xx,yy],-1)*span[j]+b[:2];anchor=pred['anchor'][j];center=(grid*anchor[...,None]).sum((0,1))/anchor.sum();unit=UNIT[ids[j]];record=dict(index=r['index'],episode=r['episode'],map_name=r['map_name'],subset=r['subset'],instruction=r['instruction'],word=r['word'],landmark=r['landmark_name'],target_type=labels[r['index']]['target_type'],target_world=targets[j].tolist(),anchor_mean=center.tolist(),gt_direction_margin=float((targets[j]-center)@unit),anchor_extent_status=interval_status(np.concatenate(r['public_polygons']),targets[j],unit)[0],methods={})
  for m in METHODS:
   pr=pred[m][j];mean=(grid*pr[...,None]).sum((0,1))/pr.sum();peak=grid.reshape(-1,2)[pr.argmax()];delta=mean-center;dot=float(delta@unit);cos=dot/max(np.linalg.norm(delta),1e-8);ne=float(np.linalg.norm(peak-targets[j]));dist=np.linalg.norm(grid-targets[j],axis=-1);record['methods'][m]=dict(peak=peak.tolist(),mean=mean.tolist(),NE=ne,R20=bool(ne<=20),GT20_mass=float(pr[dist<=20].sum()),direction_sign=bool(dot>0),direction_sector45=bool(cos>=2**-.5))
  record['direction_sector_correct_but_not_R20']=record['methods']['direction_real']['direction_sector45'] and not record['methods']['direction_real']['R20'];details.append(record)
 diagnostic={}
 for sub in ['fit','dev']:
  rr=[r for r in details if r['subset']==sub];miss=[r for r in rr if r['direction_sector_correct_but_not_R20']];diagnostic[sub]=dict(n=len(rr),maps=len({r['map_name'] for r in rr}),unique_episodes=len({r['episode'] for r in rr}),model_sign_correct=sum(r['methods']['direction_real']['direction_sign'] for r in rr),model_sector_correct=sum(r['methods']['direction_real']['direction_sector45'] for r in rr),sector_correct_but_missed=len(miss),sector_correct_but_missed_GT_same_halfplane=sum(r['gt_direction_margin']>0 for r in miss),GT_same_halfplane=sum(r['gt_direction_margin']>0 for r in rr),GT_extent=dict(Counter(r['anchor_extent_status'] for r in rr)))
 (o/'metrics.json').write_text(json.dumps(metrics,indent=2));(o/'diagnostics.json').write_text(json.dumps(diagnostic,indent=2));(o/'scored_samples.json').write_text(json.dumps(details,ensure_ascii=False,indent=2));(o/'scoring_manifest.json').write_text(json.dumps(dict(labels_sha256=file_sha256(a.labels),prediction_sha256=seal['prediction_sha256'],sealed_predictions_unchanged=file_sha256(o/'probabilities.npz')==seal['prediction_sha256'],selection_unchanged=file_sha256(o/'selection.json')==seal['selection_sha256'],ground_truth_used_only_in_this_posthoc_process=True,producer_sha256=file_sha256(__file__)),indent=2))
 with (o/'samples.csv').open('w') as f:
  keys=['index','episode','subset','word','landmark','gt_direction_margin','anchor_extent_status'];fields=keys+[m+'_'+x for m in METHODS for x in ['R20','NE','GT20_mass']];w=csv.DictWriter(f,fields);w.writeheader()
  for r in details:w.writerow({**{k:r[k] for k in keys},**{m+'_'+k:r['methods'][m][k] for m in METHODS for k in ['R20','NE','GT20_mass']}})
 # All retained Dev cases are shown, plus the first fit case per available word;
 # GT is used for evaluation overlays, never model inputs or method selection.
 chosen=[j for j,r in enumerate(rows) if r['subset']=='dev']
 for word in ['north','south']:
  chosen.append(next(j for j,r in enumerate(rows) if r['subset']=='fit' and r['word']==word))
 cache={};images=[]
 def background(mp):
  if mp not in cache:
   with rasterio.open(a.data_root/'rgbd'/f'{mp}.tif') as t:ext=[t.bounds.left,t.bounds.right,t.bounds.bottom,t.bounds.top]
   cache[mp]=(cv2.cvtColor(cv2.imread(str(a.data_root/'rgbd'/f'{mp}.png')),cv2.COLOR_BGR2RGB),ext)
  return cache[mp]
 for j in chosen:
  r=rows[j];rr=details[j];bg,ext=background(r['map_name']);b=bounds[j];fig,axes=plt.subplots(1,4,figsize=(17,4.8))
  for ax,m in zip(axes,['anchor','nodirection_real','direction_marginal','direction_real']):
   ax.imshow(bg,extent=ext);pr=pred[m][j];ax.imshow(pr,origin='lower',extent=[b[0],b[2],b[1],b[3]],cmap='magma',alpha=np.where(pr>pr.max()*.02,.55,0),vmin=0,vmax=pr.max())
   for poly in r['public_polygons']:
    pp=np.array(poly);ax.plot(*np.vstack([pp,pp[0]]).T,'c-',lw=.9)
   ax.scatter(*targets[j],c='lime',marker='*',s=75);ax.scatter(*rr['methods'][m]['peak'],c='red',marker='x',s=35);center=np.array(rr['anchor_mean']);ax.annotate(r['word']+' (engineering)',xy=center+25*UNIT[ids[j]],xytext=center,color='yellow',fontsize=7,arrowprops=dict(arrowstyle='->',color='yellow'));ax.set(xlim=(b[0],b[2]),ylim=(b[1],b[3]),title=f"{m}\nNE={rr['methods'][m]['NE']:.1f}m mass20={rr['methods'][m]['GT20_mass']:.3f}")
  fig.suptitle(f"Episode {r['episode']} | {r['subset']} | {r['word']} of {r['landmark_name']}\nCyan public anchor; green TRAIN target GT; red predicted peak; +Y up/+X right, geographic compass unverified",fontsize=9);fig.tight_layout();fn=f"case_{r['index']:03d}_{r['episode']}.png";fig.savefig(o/fn,dpi=110);plt.close(fig);images.append(dict(index=r['index'],episode=r['episode'],image=fn))
 for j in [chosen[-2],chosen[0]]:
  r=rows[j];b=bounds[j];bg,ext=background(r['map_name']);fig,axes=plt.subplots(1,4,figsize=(17,4))
  for k,ax in enumerate(axes):
   ax.imshow(bg,extent=ext);pr=pred['counterfactual'][j,k];ax.imshow(pr,origin='lower',extent=[b[0],b[2],b[1],b[3]],cmap='magma',alpha=np.where(pr>pr.max()*.02,.6,0))
   for poly in r['public_polygons']:
    q=np.array(poly);ax.plot(*np.vstack([q,q[0]]).T,'c-',lw=.7)
   ax.scatter(*targets[j],c='lime',marker='*');ax.set(title=['north','south','east','west'][k],xlim=(b[0],b[2]),ylim=(b[1],b[3]))
  fig.suptitle(f"Episode {r['episode']} fixed public Anchor; only word ID changes. Green is original GT, not GT for hypothetical words.",fontsize=9);fig.tight_layout();fig.savefig(o/f'word_control_{r["index"]}.png',dpi=110);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4));names=METHODS;vv=metrics['dev']
 for ax,key,label in zip(axes,['R20','NE','GT20_mass'],['R@20m','NE (m)','GT20 probability mass']):ax.bar(names,[vv[m]['all'][key] for m in names]);ax.set_title(label);ax.tick_params(axis='x',rotation=45,labelsize=7)
 fig.suptitle('Retained previous TRAIN Dev only; no fresh independent confirmation');fig.tight_layout();fig.savefig(o/'summary.png',dpi=140);plt.close(fig)
 (o/'cases.json').write_text(json.dumps(images,indent=2));page='<meta charset="utf-8"><title>严格直接方向Belief</title><style>body{font:16px sans-serif;margin:20px}img{max-width:100%}</style><h1>严格直接关系冻结专项诊断</h1><p>拟合数据与此前Dev分开；无新训练，无地理方向校正。绿色仅后验GT。</p><a href="REPORT_zh.md">中文报告</a><img src="summary.png">'
 for c in images:
  r=next(x for x in details if x['index']==c['index']);page+=f'<h2>{r["episode"]} {r["subset"]}</h2><p>{html.escape(r["instruction"])}</p><a href="{c["image"]}"><img loading="lazy" src="{c["image"]}"></a>'
 (o/'index.html').write_text(page);print(json.dumps(diagnostic,indent=2))
if __name__=='__main__':main()
