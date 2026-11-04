"""Render immutable predictions on real TRAIN orthophotos; GT is display-only."""
import argparse,csv,html,json
from pathlib import Path
import cv2,numpy as np,rasterio,shapely
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from graphnav.polygon_direction_region import UNITS,polygon_direction_region
from graphnav.checkpointing import file_sha256

METHODS=['anchor','centroid32','no_direction','original32','polygon_ray','polygon_cone']
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();o=a.directory;public=json.loads((o/'public.json').read_text());byindex={r['index']:r for r in public};details=sum([json.loads((o/f'{phase}_samples.json').read_text()) for phase in ['development','confirmation']],[]);byid={r['index']:r for r in details};metrics={**json.loads((o/'development_metrics.json').read_text()),**json.loads((o/'confirmation_metrics.json').read_text())};pred=np.load(o/'predictions.npz');seal=json.loads((o/'prediction_seal.json').read_text());assert file_sha256(o/'predictions.npz')==seal['prediction_sha256']
 cache={}
 def background(mp):
  if mp not in cache:
   with rasterio.open(a.data_root/'rgbd'/f'{mp}.tif') as t:extent=[t.bounds.left,t.bounds.right,t.bounds.bottom,t.bounds.top]
   cache[mp]=(cv2.cvtColor(cv2.imread(str(a.data_root/'rgbd'/f'{mp}.png')),cv2.COLOR_BGR2RGB),extent)
  return cache[mp]
 # Show every confirmation case, not only positive examples. All rare east/west
 # tasks are explicitly training/exposure-group illustrations, not held-out tests.
 chosen=[r['index'] for r in public if r['subset']=='confirm' or r['word'] in ['east','west']]
 chosen.extend([r['index'] for r in public if r['subset']=='dev'][:4]);cases=[]
 for j in chosen:
  r=byindex[j];d=byid[j];b=np.array(r['bounds']);bg,extent=background(r['map_name']);gt=np.array(d['target_world']);fig,axes=plt.subplots(1,5,figsize=(22,5))
  for ax,name in zip(axes,['anchor','centroid32','original32','polygon_ray','polygon_cone']):
   pr=pred[name][j];ax.imshow(bg,extent=extent);ax.imshow(pr,origin='lower',extent=[b[0],b[2],b[1],b[3]],cmap='magma',alpha=np.where(pr>pr.max()*.01,.6,0),vmin=0,vmax=pr.max())
   for poly in r['public_polygons']:
    xy=np.array(poly);ax.plot(*np.vstack([xy,xy[0]]).T,'cyan',lw=.8)
   ax.scatter(*gt,c='lime',marker='*',s=65);ax.add_patch(Circle(gt,20,fill=False,color='lime',lw=.8));ax.scatter(*d['methods'][name]['peak'],c='red',marker='x',s=35);ax.set(xlim=(b[0],b[2]),ylim=(b[1],b[3]),title=f"{name}\nNE={d['methods'][name]['NE']:.1f}m, mass20={d['methods'][name]['GT20_mass']:.3f}",xlabel='World X',ylabel='World Y')
  fig.suptitle(f"Episode {r['episode']} | {r['subset']} | {r['word']} of {r['landmark_name']}\nCyan public Polygon; green original TRAIN GT/20m circle; red peak. Engineering +Y up/+X right; compass meaning unverified.",fontsize=10);fig.tight_layout();filename=f"case_{r['index']:03d}_{r['episode']}.png";fig.savefig(o/filename,dpi=120);plt.close(fig);cases.append(dict(index=j,episode=r['episode'],subset=r['subset'],word=r['word'],image=filename))
 # Fixed public example, all four words, no new target labels for hypothetical words.
 r=public[0];bg,extent=background(r['map_name']);b=np.array(r['bounds']);fig,axes=plt.subplots(1,4,figsize=(17,4))
 for ax,word in zip(axes,['north','south','east','west']):
  out=polygon_direction_region(r['public_polygons'],r['bounds'],word);pr=out['probabilities'];ax.imshow(bg,extent=extent);ax.imshow(pr,origin='lower',extent=[b[0],b[2],b[1],b[3]],cmap='magma',alpha=np.where(pr>max(pr.max()*.01,0),.65,0));
  for poly in r['public_polygons']:
   xy=np.array(poly);ax.plot(*np.vstack([xy,xy[0]]).T,'c-',lw=.8)
  ax.set(title=f'{word}: {out["support"].sum()} cells',xlim=(b[0],b[2]),ylim=(b[1],b[3]))
 fig.suptitle('Same public full Polygon, only direction changes. No target GT is supplied.');fig.tight_layout();fig.savefig(o/'four_direction_regions.png',dpi=120);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,key in zip(axes,['R20','NE','GT20_mass']):
  v=metrics['confirm'];ax.bar(METHODS,[v[n]['all'][key] for n in METHODS]);ax.set_title(key+' | new confirmation n=30');ax.tick_params(axis='x',rotation=45,labelsize=7)
 fig.tight_layout();fig.savefig(o/'confirmation_summary.png',dpi=140);plt.close(fig)
 # Additional descriptive diagnosis of the frozen confirmation; no rule is changed.
 confirm=[r for r in details if r['subset']=='confirm'];inside=[]
 for r in confirm:
  pub=byindex[r['index']];region=unary_union([shapely.Polygon(x) for x in pub['public_polygons']]);inside.append(dict(index=r['index'],episode=r['episode'],target_inside_public_anchor=bool(region.covers(shapely.Point(r['target_world']))),public_types=pub['public_types']))
 rng=np.random.default_rng(20261104);groups=sorted({r['group'] for r in confirm});deltas=[]
 for _ in range(5000):
  sample=[r for g in rng.choice(groups,len(groups),replace=True) for r in confirm if r['group']==g];deltas.append([np.mean([float(r['methods']['polygon_cone']['R20'])-float(r['methods']['original32']['R20']) for r in sample]),np.mean([r['methods']['polygon_cone']['NE']-r['methods']['original32']['NE'] for r in sample])])
 ci=np.quantile(deltas,[.025,.975],axis=0);diagnosis=dict(confirmation_anchor_overlap=inside,inside_count=sum(r['target_inside_public_anchor'] for r in inside),support_and_missed_count=sum(r['methods']['polygon_cone']['legal_support_GT20'] and not r['methods']['polygon_cone']['R20'] for r in confirm),paired_group_bootstrap=dict(groups=len(groups),resamples=5000,seed=20261104,R20_delta_CI95=ci[:,0].tolist(),NE_delta_CI95=ci[:,1].tolist(),scope='Exploratory, 13 groups, maps overlap; not a geographic-generalization significance claim'))
 (o/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2));(o/'cases.json').write_text(json.dumps(cases,indent=2))
 with (o/'samples.csv').open('w') as f:
  cols=['index','candidate_id','episode','subset','word','landmark_name','gt_engineering_direction_margin','gt_polygon_axis_status'];names=cols+[m+'_'+k for m in METHODS for k in ['R20','NE','GT20_mass']];w=csv.DictWriter(f,names,lineterminator="\n");w.writeheader()
  for r in details:w.writerow({**{k:r[k] for k in cols},**{m+'_'+k:r['methods'][m][k] for m in METHODS for k in ['R20','NE','GT20_mass']}})
 page='<meta charset="utf-8"><title>Polygon方向区域诊断</title><style>body{font:16px sans-serif;margin:20px}img{width:100%}article{border-top:1px solid #aaa}</style><h1>全量TRAIN扫描与完整Polygon方向区域</h1><p>新确认组全部展示；GT仅后验，非人工像素Gold。没有训练新网络。</p><a href="REPORT_zh.md">中文报告</a><img src="confirmation_summary.png"><img src="four_direction_regions.png">'
 for c in cases:
  r=byindex[c['index']];page+=f'<article><h2>{r["episode"]} | {r["subset"]} | {r["word"]}</h2><p>{html.escape(r["instruction"])}</p><a href="{c["image"]}"><img loading="lazy" src="{c["image"]}"></a></article>'
 (o/'index.html').write_text(page);assert file_sha256(o/'predictions.npz')==seal['prediction_sha256'];print(json.dumps(dict(images=len(cases),diagnosis=diagnosis['inside_count']),indent=2))
if __name__=='__main__':main()
