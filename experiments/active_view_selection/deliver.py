"""Aggregate completed, fixed-protocol coverage results and compact artifacts."""
import csv,gzip,hashlib,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from .run import R,dump,digest

A=Path('docs/assets/active_view_selection')
def read(p):return json.loads(p.read_text())
def package():
 A.mkdir(parents=True,exist_ok=True);rows=read(R/'predictions.json');extra=read(R/'oracle_extra_predictions.json');metrics=read(R/'metrics.json');execution=read(R/'execution.json')
 for split in ['train','dev','all']:
  selected=[r for r in extra if split=='all' or r['split']==split];metrics[split]['target_directed_oracle']={k:summarize_rows(selected,k,seed=20261011) for k in rows[0]['metrics']};metrics[split]['target_directed_oracle']['independent_visible_errors']=len({r['family_id'] for r in selected if r['metrics']['error_credible']})
 dev=metrics['dev'];a=dev['active_minus_fixed_nadir'];b=dev['active_minus_random_legal'];passes=bool(a['mean'] is not None and a['mean']>=.08 and a['map_cluster_95ci'][0]>0 and b['mean']>0);visible=dev['active_view']['independent_visible_errors']
 gates=dict(stage_A_pass=passes,coverage_delta_pp=None if a['mean'] is None else a['mean']*100,coverage_map_ci_pp=None if a['map_cluster_95ci'] is None else [v*100 for v in a['map_cluster_95ci']],active_minus_random_pp=None if b['mean'] is None else b['mean']*100,independent_visible_errors=visible,stage_B_run=False,stage_B_coverage_pass=visible>=50,training=False,closedloop=False,unseen=False,decision='stop after coverage diagnosis' if not passes else 'coverage pilot gain; insufficient50 independent visible errors, no identity acceptance')
 dump(R/'acceptance.json',gates);dump(R/'combined_metrics.json',metrics);dump(A/'metrics.json',metrics)
 for name in ['execution.json','oracle_extra_execution.json','cohort_manifest.json','causal_replay.json','geometry_checks.json','planner_audit.json','initial_attempt.json','acceptance.json','oracle_catalogue.json']:
  shutil.copy2(R/name,A/name)
 for name in ['predictions.json','oracle_extra_predictions.json']:
  with gzip.open(A/(name+'.gz'),'wt') as f:json.dump(read(R/name),f,allow_nan=False)
 framemeta=[read(p) for p in sorted((R/'frames').glob('*/metadata.json'))]
 with gzip.open(A/'all_frame_metadata.json.gz','wt') as f:json.dump(framemeta,f,allow_nan=False)
 for name in ['protocol.json','oracle_extras_protocol.json']:shutil.copy2(Path(__file__).with_name(name),A/name)
 shutil.copy2(R/'regression_tests.log',A/'regression_tests.txt')
 columns=['split','strategy','n','target_fov','target_credible','candidate_coverage','xy_m','z_m','gimbal_degrees','extra_observations','max_native_target_side','inside_hull_valid']
 with (A/'comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');w.writeheader()
  for split in ['train','dev','all']:
   for method,v in metrics[split].items():
    if method.startswith('active_minus'):continue
    w.writerow(dict(split=split,strategy=method,n=v['target_credible']['n'],**{k:v[k]['mean'] for k in columns[3:]}))
 labels=['fixed_nadir','camera_only','move_nadir','active_view','random_legal','existing_belief_active','view_oracle','target_directed_oracle'];fig,ax=plt.subplots(figsize=(10,4));means=[metrics['all'][v]['target_credible']['mean']*100 for v in labels];ci=np.array([metrics['all'][v]['target_credible']['map_cluster_95ci'] for v in labels])*100;x=np.arange(len(labels));ax.bar(x,means,color=['gray','blue','blue','green','gray','gray','orange','orange']);ax.errorbar(x,means,yerr=np.maximum(0,np.stack([np.array(means)-ci[:,0],ci[:,1]-means])),fmt='none',ecolor='black',capsize=3);ax.set(xticks=x,xticklabels=labels,ylabel='Credible target coverage (%)',ylim=(0,110),title='Exploratory Train + internal Dev; map bootstrap CI');ax.tick_params(axis='x',labelrotation=25);fig.tight_layout();fig.savefig(A/'coverage.png',dpi=150);plt.close(fig)
 # Actual current/reached RGB, target bbox drawn only for offline diagnostics.
 chosen=sorted([r for r in rows if r['strategy']=='active_view'],key=lambda r:(-r['metrics']['target_credible'],r['sample_id']))[:3]
 for i,r in enumerate(chosen):
  alternatives=[next(x for x in rows if x['sample_id']==r['sample_id'] and x['strategy']==m) for m in ['fixed_nadir','camera_only','move_nadir','active_view']]+[next(x for x in extra if x['sample_id']==r['sample_id'])];canvas=Image.new('RGB',(5*256,560),'white');d=ImageDraw.Draw(canvas)
  for j,c in enumerate(alternatives):
   d.text((j*256+4,4),c['strategy'],fill='black')
   for step,e in enumerate(c['exposures']):
    img=Image.open(R/'frames'/e['frame_id']/'rgb.png').convert('RGB');draw=ImageDraw.Draw(img);reg=e['diagnostic']['target_region']
    if reg:draw.rectangle(reg['bbox'],outline='red',width=3)
    canvas.paste(img.resize((256,256)),(j*256,25+step*265));d.text((j*256+3,25+step*265),f"tilt {e['view']['camera_tilt']} / credible {e['diagnostic']['credible']}",fill='yellow')
  canvas.save(A/f'case_{i+1}.png')
 # Quality failures reclassified transparently from fixed raw ROI/support fields.
 failures={}
 for method in labels:
  selected=[r for r in rows+extra if r['strategy']==method];counts=Counter()
  for r in selected:
   for e in r['exposures']:
    reason=e['diagnostic']['reason'];reg=e['diagnostic']['target_region'];meta=next(m for m in framemeta if m['frame_id']==e['frame_id'])
    if reason=='native_pixels_or_support':reason='native_pixels' if reg['native_min_side']<16 or reg['measured_visible_pixels']<64 else 'roi_point_support_holes'
    elif reason=='full_frame_quality':reason='frame_internal_holes' if meta['inside_hull_valid']<.6 else 'frame_extent_or_scan_boundary'
    counts[reason]+=1
  failures[method]=dict(counts)
 dump(A/'failure_reasons.json',failures)
 dump(A/'resource_audit.json',dict(unique_rendered_views=len(framemeta),actual_unique_frame_render_seconds=sum(x['render_seconds'] for x in framemeta),main_wall_seconds=execution['seconds'],initial_attempt_wall_seconds_approx=read(R/'initial_attempt.json')['wall_seconds_approx'],oracle_extra_wall_seconds=read(R/'oracle_extra_execution.json')['seconds'],original_objects_sha256=digest(Path('/home/rental/20260922_1/Workspace/DATA/sbfnav-data/cityrefer/objects.json')),clouds={k:dict(v,sha256=digest(Path(v['path']))) for k,v in execution['clouds'].items()},no_GPU_encoding=True))
 dump(A/'sha256.json',{str(p.relative_to(A)):digest(p) for p in sorted(A.iterdir()) if p.name!='sha256.json'})
 print(json.dumps(gates),flush=True)

if __name__=='__main__':package()
