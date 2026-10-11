"""Package compact reproducible P3 audit outputs; raw data/weights remain local."""
import gzip,hashlib,json,shutil,csv
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
R=Path('runs/multiview_rgb_grounding'); A=Path('docs/assets/multiview_rgb_grounding');A.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def save(name,x):(A/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
quality={};qrows=[]
for folder in ['quality_final','quality_dev_error_probe']:
 quality.update(read(R/folder/'quality_metrics.json'));qrows+=read(R/folder/'quality_predictions.json')
save('quality_metrics.json',quality)
with gzip.open(A/'quality_predictions.json.gz','wt') as f:json.dump(qrows,f,allow_nan=False)
for name in ['geometry_baseline_equivalence.json','renderer_equivalence_real.json','cuda_smoke.json','coverage_geometric_proxy.json']:
 shutil.copy2(R/name,A/name)
for name in ['coverage_geometric_proxy_predictions.json']:
 with gzip.open(A/(name+'.gz'),'wt') as f:json.dump(read(R/name),f)
comparisons=[]
for split,folder in [('train','matching_train_pilot_world'),('dev','matching_dev_error_probe')]:
 metrics=read(R/folder/'metrics.json');save(split+'_matching_metrics.json',metrics);save(split+'_coverage.json',read(R/folder/'coverage.json'))
 rows=read(R/folder/'predictions.json')
 with gzip.open(A/(split+'_predictions.json.gz'),'wt') as f:json.dump(rows,f)
 for key,v in metrics['summary'].items():
  model,lang,scope,modality,control,view=key.split('/')
  comparisons.append(dict(split=split,model=model,language=lang,scope=scope,modality=modality,control=control,view=view,n=v['instance_r1']['n'],r1=v['instance_r1']['mean'],hard_r1=v['hard_r1']['mean']))
with (A/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(comparisons[0]),lineterminator="\n");w.writeheader();w.writerows(comparisons)
fig,ax=plt.subplots(figsize=(8,4))
for name,v in quality.items():ax.plot([0,20,40,60],[v['per_tilt'][str(t)]['frame_quality_pass']['mean']*100 for t in [0,20,40,60]],marker='o',label=name.replace('render_',''))
ax.axhline(80,ls='--',color='gray',label='pilot gate 80%');ax.set(xlabel='Tilt from nadir (degrees)',ylabel='Frame quality pass (%)',ylim=(0,105));ax.legend(fontsize=7);fig.tight_layout();fig.savefig(A/'quality.png',dpi=160);plt.close(fig)
# A real measured-point 3D RGB/depth/visibility contact sheet for a Dev error case.
manifest=read(R/'render_dev_error_probe/manifest.json');frames=[v for v in manifest['frames'] if '5593' in v['metadata']['sample_id']];frames=sorted(frames,key=lambda x:x['tilt'])
canvas=Image.new('RGB',(1024,3*256+40),(255,255,255));draw=ImageDraw.Draw(canvas)
for j,v in enumerate(frames):
 base=R/'render_dev_error_probe'/v['frame_key'];rgb=Image.open(base/f"tilt{v['tilt']}_r1_rgb.png").convert('RGB');geo=np.load(base/f"tilt{v['tilt']}_r1_geometry.npz");depth=geo['depth'];valid=np.isfinite(depth);scaled=np.zeros((*depth.shape,3),np.uint8)
 if valid.any():
  lo,hi=np.percentile(depth[valid],[2,98]);c=plt.get_cmap('viridis')(np.clip((depth-lo)/max(hi-lo,1e-6),0,1))[...,:3];scaled[valid]=(c[valid]*255).astype(np.uint8)
 for k,img in enumerate([rgb,Image.fromarray(scaled),Image.fromarray((valid*255).astype('uint8')).convert('RGB')]):canvas.paste(img.resize((256,256)),(j*256,40+k*256))
 draw.text((j*256+5,5),f"tilt {v['tilt']} / valid {v['valid_fraction']:.2f}",fill='black')
canvas.save(A/'dev_rgb_depth_visibility.png')
shutil.copy2(R/'early_train_view_montage.png',A/'train_perspective.png')
# Only metadata aggregates: no external copyrighted README/image/caption redistribution.
resources=read(R/'resource_audit/audit.json');save('resource_repositories.json',resources['repositories'])
for name in ['probes_corrected.json','aerialvln_archive_audit.json']:
 shutil.copy2(R/'resource_audit'/name,A/name)
save('audit_summary.json',dict(rendered_frames=sum(len(read(R/p/'manifest.json')['frames']) for p in quality),poses=len({r['sample_id'] for r in qrows}),independent_families=len({r['family_id'] for r in qrows}),credible_visible_error_families=len({r['family_id'] for r in qrows if r['geometry_error'] and r['metrics']['target_credible_visible']}),failure_reasons=dict(Counter(r['target_failure_reason'] for r in qrows)),training=0,seen=False,unseen=False,goal_fusion=False,scope='bounded Train/internal Dev exploratory pilot; formal P3-B quality gate failed'))
save('artifact_sha256.json',{str(p.relative_to(A)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(A.iterdir()) if p.name!='artifact_sha256.json'})
print((A/'audit_summary.json').read_text())
