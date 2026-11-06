"""Figures from actual TRAIN statistics and actual map rasters; no synthetic outcomes."""
import argparse,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.transforms import Affine2D
from PIL import Image
import rasterio
import pandas as pd
from common import *
from infer import region,anchor
BASE=Path(__file__).parent;P=BASE/'results';F=BASE/'figures';F.mkdir(exist_ok=True);Image.MAX_IMAGE_PIXELS=None
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':320,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COL={'strict_rules':'#009E73','semantic_candidates':'#D55E00','bert_router':'#0072B2','shuffled_label_router':'#777777','bert_support_uniform_orientation':'#BBBBBB','semantic_no_road_facing_prior':'#E69F00','all_geometry_candidates':'#CC79A7'}
def save(fig,name):fig.savefig(F/(name+'.png'),dpi=320,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def run(root):
 S=readj(P/'summary.json');null=readj(P/'negative_controls.json');B=readj(P/'breakdown.json');C=readj(P/'dataset.json');cov=readj(P/'frame_coverage.json');X={r['occurrence_id']:r for r in readlines(P/'public_inputs.jsonl.gz')};PP={r['occurrence_id']:r for r in readlines(P/'inference_predictions.jsonl.gz')};E={r['occurrence_id']:r for r in readlines(P/'evaluation_gt.jsonl.gz')};outputs=readlines(P/'reference_frame_outputs.jsonl.gz');ev=readlines(P/'evaluation_predictions.jsonl.gz')
 def summary(c,m,scheme='object_5fold',mode='conditional_local'):return next(s for s in S if s['cohort']==c and s['model']==m and s['scheme']==scheme and s['mode']==mode)
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');labels=['v2 all','v2 high','expanded all','expanded high'];ax.bar(labels,[C[k] for k in ['v2_disjoint','v2_high_disjoint','expanded_disjoint','expanded_high_disjoint']],color=['#777777','#AAAAAA','#0072B2','#56B4E9']);ax.set(ylabel='Disjoint polygon pairs (all requested direction words)',title='Original v2 pairs preserved exactly; typed-road expansion is text-only')
 for i,k in enumerate(['v2_disjoint','v2_high_disjoint','expanded_disjoint','expanded_high_disjoint']):ax.text(i,C[k]+4,str(C[k]),ha='center')
 save(fig,'cohort_coverage')
 fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
 for model in ['strict_rules','semantic_candidates','bert_router','all_geometry_candidates','shuffled_label_router']:
  z=summary('v2_all',model);x=z['coverage'];y=z['consistency'];lo,hi=z.get('wilson95_if_binary',z['cluster95'])
  ax.errorbar(x,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',color=COL[model],capsize=4,label=f"{model}: {z['covered']}/{z['n']}")
 ax.set(xlim=(-.03,1.04),ylim=(0,1.06),xlabel='Coverage among v2 core pairs',ylabel='Conditional sector compatibility',title='Hard heading success and mixture compatibility are different output types');ax.legend(fontsize=8);save(fig,'coverage_conditional_compatibility')
 fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 for ax,scheme in zip(axs,['object_5fold','leave_map_out']):
  cohorts=['v2_all','v2_high','expanded_all','expanded_high']
  for j,m in enumerate(['strict_rules','semantic_candidates','bert_router','shuffled_label_router']):
   for i,c in enumerate(cohorts):
    z=summary(c,m,scheme);y=z['net_improvement_all'];lo,hi=z['net_cluster95'];ax.errorbar(i+(j-1.5)*.13,100*y,yerr=[[100*max(0,y-lo)],[100*max(0,hi-y)]],fmt='o',capsize=3,color=COL[m],label=m if i==0 else None)
  ax.axhline(0,color='gray',linestyle=':');ax.set(xticks=range(4),xticklabels=cohorts,ylabel='Paired hybrid compatibility gain (percentage points)',title=scheme);ax.tick_params(axis='x',rotation=15)
 axs[0].legend(fontsize=7);save(fig,'net_gain_crossvalidation')
 fig,ax=plt.subplots(figsize=(9,5),layout='constrained');controls=[z for z in null if z['cohort']=='v2_all' and z['model']=='bert_router']
 for i,z in enumerate(controls):
  y=z['null_mean'];lo,hi=z['null95'];ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',capsize=5,color='#777777',label='fixed-coverage randomization 95% range' if i==0 else None);ax.scatter(i,z['observed_conditional'],marker='*',s=130,color=COL['bert_router'],label='observed candidate compatibility' if i==0 else None)
 ax.set(xticks=range(len(controls)),xticklabels=[z['control'].replace('_','\n') for z in controls],ylim=(0,.6),ylabel='Conditional sector compatibility',title='v2 all: BERT candidate support held fixed');ax.legend(fontsize=8);save(fig,'geometry_negative_controls')
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');quality=readj(P/'router_quality_silver.json')
 for j,m in enumerate(['bert_router','shuffled_label_router']):
  q=[z for z in quality if z['model']==m];ax.bar(np.arange(2)+(j-.5)*.25,[z['silver_label_macroF1'] for z in q],width=.25,label=m,color=COL[m])
 ax.set(xticks=range(2),xticklabels=['object_5fold','leave_map_out'],ylim=(0,1),ylabel='Macro F1 vs text-rule SILVER labels',title='NOT accuracy of identifying the true author reference frame');ax.legend();save(fig,'bert_silver_label_control')
 fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
 for i,w in enumerate(['left','right','front','behind']):
  z=next(q for q in B if q['scheme']=='object_5fold' and q['cohort']=='v2_all' and q['model']=='bert_router' and q['group_type']=='word' and q['group']==w);y=z['net_improvement_all'];lo,hi=z['net_cluster95'];ax.errorbar(i,100*y,yerr=[[100*max(0,y-lo)],[100*max(0,hi-y)]],fmt='o',capsize=5,color=COL['bert_router']);ax.annotate(f"candidate support {z['covered']}/{z['n']}",(i,100*y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
 ax.axhline(0,color='gray',linestyle=':');ax.set(xticks=range(4),xticklabels=['left','right','front','behind'],ylabel='Paired compatibility gain (percentage points)',title='v2 all, object-group crossfit; unknown cases keep world baseline');save(fig,'per_word_gain')
 core=[r for r in outputs if r['word'] in MAIN];counts=collections.Counter(r['rule_state'] for r in core);fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');names=sorted(counts);ax.barh(names,[counts[n] for n in names],color=['#E69F00' if 'candidate' in n else '#009E73' if 'explicit' in n else '#999999' for n in names]);ax.set(xlabel='All named-landmark core relation occurrences',title='Single heading availability is NOT candidate-family coverage')
 for i,n in enumerate(names):ax.text(counts[n]+4,i,f'{counts[n]} ({counts[n]/len(core):.1%})',va='center')
 save(fig,'unknown_and_candidates')
 # Select illustrative real cases only after predictions are frozen. Never feed
 # case selection back into the rules/router or tune the source direction with GT.
 chosen=[('cambridge_block_28:8:2:142','known_boathouse'),('cambridge_block_28:165:2:32','new_route_agreement'),('cambridge_block_28:174:5:26','sector_vs_roadside'),('cambridge_block_33:53:3:54','new_map_roadside'),('cambridge_block_21:95:3:42','new_junction_roadside'),('cambridge_block_28:17:3:26','route_proxy_conflict'),('cambridge_block_13:20:3:24','observer_region_uncertainty')]
 unknown=next((rid for rid in sorted(X) if PP[rid]['family_silver']=='intrinsic_candidate' and not PP[rid]['strict_ids'] and E[rid]['geometry_status']=='disjoint' and len([c for c in PP[rid]['candidates'] if c['type']=='building_PCA_axis'])==4),None)
 if unknown:chosen.append((unknown,'unsigned_building_unknown'))
 caseindex=[]
 for k,(rid,label) in enumerate(chosen):
  if rid not in X:continue
  inp=X[rid];pr=PP[rid];e=E[rid];m=inp['map'];rp=region(inp,inp['reference_ids']);rxy=np.array(e['reference_xy']);target=np.array(e['target_polygon']);subxy=np.array(e['subject_xy']);c=None
  if pr['strict_ids']:c=next(x for x in pr['candidates'] if x['candidate_id'] in pr['strict_ids'])
  else:
   c=next((x for x in pr['candidates'] if x['type']=='explicit_view_from_named_region'),None) or next((x for x in pr['candidates'] if x['type']=='building_PCA_axis'),None)
  h=c['heading_deg'] if c else None;points=np.vstack([vertices(rp),target]);q=None
  if c:
   q=c['meta'].get('junction_xy') or c['meta'].get('water_boundary_point') or c['meta'].get('observer_source_xy')
   if q:points=np.vstack([points,q])
  pad=max(10,float(max(points.max(0)-points.min(0))*.08));mn=points.min(0)-pad;mx=points.max(0)+pad
  with rasterio.open(root/'rgbd'/f'{m}.tif') as src:
   tr=src.transform;c0=max(0,int(np.floor((mn[0]-tr.c)/tr.a)));c1=min(src.width,int(np.ceil((mx[0]-tr.c)/tr.a)));r0=max(0,int(np.floor((mx[1]-tr.f)/tr.e)));r1=min(src.height,int(np.ceil((mn[1]-tr.f)/tr.e)))
   with Image.open(root/'rgbd'/f'{m}.png') as img:patch=np.array(img.crop((c0,r0,c1,r1)))
   extent=[tr.c+c0*tr.a,tr.c+c1*tr.a,tr.f+r1*tr.e,tr.f+r0*tr.e]
  fig=plt.figure(figsize=(14,10),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=[4,1.15]);contextids=[i for i in (c['source_ids'] if c else []) if i not in inp['reference_ids']]
  for j in range(2):
   ax=fig.add_subplot(grid[0,j]);rotdeg=0 if j==0 or h is None else 90-h;a=np.deg2rad(rotdeg);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);origin=np.zeros(2) if j==0 else rxy
   convert=lambda v:(np.asarray(v)-origin)@rot.T
   transform=Affine2D().translate(-origin[0],-origin[1]).rotate_deg(rotdeg)+ax.transData;ax.imshow(patch,extent=extent,origin='upper',transform=transform)
   def draw(points,col,lab,ls='-'):
    z=convert(points);z=np.vstack([z,z[0]]);ax.plot(z[:,0],z[:,1],color=col,linewidth=2,linestyle=ls,label=lab)
   for part in (list(rp.geoms) if rp.geom_type=='MultiPolygon' else [rp]):draw(part.exterior.coords,'#0072B2','Named landmark polygon')
   draw(target,'#D55E00','Relation subject polygon')
   for oid in contextids:
    if oid in inp['scene']:draw(inp['scene'][oid]['contour'],'#009E73','Independent named/source region','--')
   z=convert([subxy])[0];o=convert([rxy])[0];ax.scatter(*z,c='#D55E00',marker='x',s=55,label='Subject GT (offline only)');ax.scatter(*o,c='#0072B2',s=25)
   if q:
    qq=convert([q])[0];ax.scatter(*qq,c='#CC79A7',marker='+',s=80,label='Source junction / region point')
   lo=convert([[extent[0],extent[2]],[extent[0],extent[3]],[extent[1],extent[2]],[extent[1],extent[3]]]);xmin,ymin=lo.min(0);xmax,ymax=lo.max(0);ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax));ax.set_aspect('equal');ax.grid(alpha=.2)
   if h is not None:
    aa=np.deg2rad(h+rotdeg);vec=np.array([np.cos(aa),np.sin(aa)]);length=max(12,min(35,max(mx-mn)*.22));ax.annotate('',xy=o+length*vec,xytext=o,arrowprops={'arrowstyle':'->','color':'#009E73','lw':3})
    if j==1 and pr['strict_ids']:
     rad=max(xmax-xmin,ymax-ymin)*1.2
     for start,end,name,color in [(45,135,'front','#009E73'),(135,225,'left','#E69F00'),(225,315,'behind','#CC79A7'),(315,405,'right','#56B4E9')]:
      ax.add_patch(Wedge((0,0),rad,start,end,facecolor=color,alpha=.075,edgecolor='none'));mid=np.deg2rad((start+end)/2);pt=np.array([np.cos(mid),np.sin(mid)])*min(rad*.22,35);ax.text(*pt,name,fontsize=9,bbox={'facecolor':'white','alpha':.6,'edgecolor':'none'})
    if not pr['strict_ids']:
     for alt in [x for x in pr['candidates'] if x['type']==c['type']]:
      aa=np.deg2rad(alt['heading_deg']+rotdeg);ax.annotate('',xy=o+length*np.array([np.cos(aa),np.sin(aa)]),xytext=o,arrowprops={'arrowstyle':'->','color':'#777777','lw':1.2,'alpha':.6})
   if j==1 and 'roadside_center_xy' in e and h is not None:
    center=convert([e['roadside_center_xy']])[0];ax.axvline(center[0],color='black',linestyle=':',linewidth=1.5,label='Centerline for SIDE diagnostic')
   ax.set(title='World XY (no geographic-north claim)' if j==0 else 'Independent local proxy; sectors only if explicit' if pr['strict_ids'] else 'UNSELECTED candidate(s): heading remains unknown',xlabel='World X (m)' if j==0 else 'Right offset (m)',ylabel='World Y (m)' if j==0 else 'Forward offset (m)');handles,labels=ax.get_legend_handles_labels();dedup=dict(zip(labels,handles));ax.legend(dedup.values(),dedup.keys(),fontsize=6,loc='best')
  txt=fig.add_subplot(grid[1,:]);txt.axis('off');rec=[r for r in ev if r['scheme']=='object_5fold' and r['occurrence_id']==rid and r['model']=='strict_rules'];sector=next((r for r in rec if r['cohort']=='expanded_all'),None);side=next((r for r in rec if r['cohort']=='roadside_secondary'),None)
  result=f"Sector: {sector['local_score'] if sector else 'not evaluable'}; road SIDE: {side['local_score'] if side else 'not evaluable'}; polygon status: {e['geometry_status']}"
  if h is not None:result+=f"; local heading={h:.2f} deg; subject bearing={e['angle_world']:.2f} deg"
  if sector and sector['local_error'] is not None:result+=f"; angle error={sector['local_error']:.2f} deg"
  body=inp['instruction']+'\n'+result+f"\n{m} | relation={inp['word']} | {rid} | rule state={pr['strict_state']}\n"+(c['evidence'] if c else 'No independently selected heading')+'\nGT is shown only for offline evaluation. Source heading is fixed before GT. Rotated imagery is an orthomap, not an original annotation view.'
  txt.text(0,1,'\n'.join(textwrap.fill(s,140) for s in body.split('\n')),va='top',fontsize=9);fig.suptitle(f'{label} | named-landmark local reference frame');fname=f'case_{k:02d}_{label}';save(fig,fname)
  caseindex.append({'occurrence_id':rid,'kind':label,'figure':fname+'.png','instruction':inp['instruction'],'reference_ids':inp['reference_ids'],'subject_ids':inp['subject_ids'],'rule_state':pr['strict_state'],'selected_candidate':c,'sector_result':sector['local_score'] if sector else None,'roadside_result':side['local_score'] if side else None,'geometry_status':e['geometry_status'],'selection_for_visualization_only':True,'known_development_example':label=='known_boathouse','used_to_select_direction_with_GT':False})
 writej(P/'case_index.json',caseindex);print('PNG/PDF figures:',len(list(F.glob('*.png'))),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True,type=Path);a=ap.parse_args();run(a.data_root)
