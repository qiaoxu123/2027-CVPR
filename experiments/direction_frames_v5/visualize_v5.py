"""High-resolution true-data plots, with frozen-v4 and v5 exploratory labels."""
from v5core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import rasterio,textwrap
from shapely.geometry import Polygon
Image.MAX_IMAGE_PIXELS=None
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','savefig.dpi':320,'pdf.fonttype':42,'font.size':10})
def save(fig,name):fig.savefig(FIG/(name+'.png'),dpi=320,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def run():
 S=readj(RESULT/'summary.json');Q=readlines(RESULT/'geometry_checks.jsonl.gz');X={x['occurrence_id']:x for x in readlines(RESULT/'public_inputs.jsonl.gz')};P={p['occurrence_id']:p for p in readlines(V4/'results/frozen_frames.jsonl.gz')};P.update({p['occurrence_id']:p for p in readlines(RESULT/'new_frames.jsonl.gz')});G={g['occurrence_id']:g for g in readlines(RESULT/'evaluation_gt.jsonl.gz')};F={f['key']:f for f in readlines(RESULT/'road_fields.jsonl.gz')};FF={f['key']:f for f in readlines(RESULT/'exploratory_fields.jsonl.gz')};bq={r['occurrence_id']:r for r in Q};audit=readj(RESULT/'v4_ambiguous22_classification.json')
 def summ(c):return next(s for s in S if s['cohort']==c and s['tier']=='all')
 fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');names=['v4_original_unique44','v5_new_bound'];bottom=np.zeros(2)
 for key,col in [('consistent','#009E73'),('conflict','#D55E00'),('ambiguous_unique','#999999')]:
  vals=[summ(c)[key] for c in names];ax.bar(names,vals,bottom=bottom,label=key,color=col)
  for i,n in enumerate(vals):
   if n:ax.text(i,bottom[i]+n/2,str(n),ha='center',va='center',color='white')
  bottom+=vals
 ax.set(ylabel='Relations on phase-1 unique-field support',title='Retrospective validation: ambiguous cases retained');ax.legend();save(fig,'historical_new_denominators')
 fig,ax=plt.subplots(figsize=(10,5),layout='constrained');cs=['v5_previously_uncovered_expression','v5_old_rule_semantics_missed_binding','v5_duplicate_or_highly_similar']
 for i,c in enumerate(cs):
  s=summ(c);ax.bar(i-.24,s['n'],width=.23,label='candidate' if i==0 else None,color='#BBBBBB');ax.bar(i,s['unique'],width=.23,label='unique geometry field' if i==0 else None,color='#0072B2');ax.bar(i+.24,s['consistent'],width=.23,label='full-contour confirmed' if i==0 else None,color='#009E73')
 ax.set(xticks=range(3),xticklabels=['new expression','missed old semantics','near duplicate'],ylabel='Actual relation count',title='Novelty of language and independent geometry are different');ax.legend();save(fig,'novelty_pipeline')
 fig,ax=plt.subplots(figsize=(10,5),layout='constrained');cats=collections.Counter(r['verified_geometric_trigger'] for r in audit);ax.barh(list(cats),list(cats.values()),color='#999999');ax.set(xlabel='Ambiguous frozen v4 relations (n=22)',title='Verified geometric trigger; causal centerline error is NOT assumed');save(fig,'ambiguous_failure_types')
 comp=readj(RESULT/'method_comparisons.json');pick=['A_frozen_v4_field','B_world_crossfit_leave_map_out','C_start_to_landmark','D_initial_UAV_yaw','E_unsigned_PCA_equal_signs','F_exploratory_smoothed_geometry'];fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
 for i,m in enumerate(pick):
  s=next(c for c in comp if c['cohort']=='bound_core' and c['method']==m);y=s['confirmed_or_compatibility_all_support'];lo,hi=s['object_cluster95'];ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=4,color='#0072B2');ax.text(i,.02,f"n={s['same_support_n']}",ha='center',fontsize=8)
 ax.set(xticks=range(6),xticklabels=['v4 field','world CV','start to ref','UAV yaw','PCA mixture','F exploration'],ylim=(0,1),ylabel='Confirmation / E mixture mass, full fixed support',title='Identical contour predicate for A-E; F changes geometry only as exploration');save(fig,'same_support_methods')
 B=readj(RESULT/'grouped_results.json');rows=[r for r in B if r['cohort']=='v5_new_bound' and r['group_type']=='map'];fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
 for i,r in enumerate(rows):
  if r['conditional_consistency'] is None:continue
  y=r['conditional_consistency'];lo,hi=r['wilson95'];ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=4);ax.text(i,.05,f"{r['consistent']}/{r['scorable']}\nunique={r['unique']}",ha='center',fontsize=8)
 ax.set(xticks=range(len(rows)),xticklabels=[r['group'].replace('_block_','\n') for r in rows],ylim=(0,1.08),ylabel='Conditional consistency / Wilson95%',title='New bindings on maps, retrospective; empty geometry support stays empty');save(fig,'new_map_results')
 # Illustration selection after all predictions frozen; never used to choose a sign.
 chosen=[('cambridge_block_28:17:3:26','Abbey_conflict'),('birmingham_block_3:99:4:30','new_proceeds_down'),('birmingham_block_3:136:0:36','new_from_B_to_A'),('cambridge_block_19:51:5:89','heldout_development_map'),('cambridge_block_21:78:5:60','new_road_pair')]
 for kind in ['boundary_tolerance_only','polygon_intersects_centerline','curvature_or_endcap_full_contour_side_nonunique']:
  rr=sorted([r for r in audit if r['verified_geometric_trigger']==kind],key=lambda r:r['occurrence_id'])
  if rr:chosen.append((rr[0]['occurrence_id'],kind))
 mismatch=next((r for r in audit if not r['GT_inside_target_polygon_1m']),None)
 if mismatch:chosen.append((mismatch['occurrence_id'],'GT_polygon_disagreement'))
 chosen.append(('cambridge_block_20:121:4:91','new_road_turn_onto_from'))
 index=[]
 for k,(rid,label) in enumerate(chosen):
  if rid not in X:continue
  x=X[rid];g=G[rid];p=P[rid];q=bq[rid];f=F.get(p.get('road_field_key'));rp=region(x,x['reference_ids']);tp=np.array(g['target_polygon']);gt=np.array(g['subject_xy'])
  if rp is None:continue
  parts=list(rp.geoms) if rp.geom_type=='MultiPolygon' else [rp];verts=np.concatenate([np.array(z.exterior.coords) for z in parts]);allpts=np.vstack([verts,tp]);panels=[(allpts.min(0)-15,allpts.max(0)+15),(np.minimum(tp.min(0),gt)-20,np.maximum(tp.max(0),gt)+20)];fig,axs=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(bottom=.27,top=.9,wspace=.18)
  with rasterio.open(DATA/'rgbd'/f"{x['map']}.tif") as src:t=src.transform;width=src.width;height=src.height
  for j,(ax,(lo,hi)) in enumerate(zip(axs,panels)):
   c0=max(0,int((lo[0]-t.c)/t.a));c1=min(width,int(np.ceil((hi[0]-t.c)/t.a)));r0=max(0,int((hi[1]-t.f)/t.e));r1=min(height,int(np.ceil((lo[1]-t.f)/t.e)))
   with Image.open(DATA/'rgbd'/f"{x['map']}.png") as im:imdata=np.array(im.crop((c0,r0,c1,r1)))
   ax.imshow(imdata,extent=[t.c+c0*t.a,t.c+c1*t.a,t.f+r1*t.e,t.f+r0*t.e],origin='upper')
   def draw(poly,col,name,ls='-'):
    v=np.array(poly);v=np.vstack([v,v[0]]);ax.plot(*v.T,c=col,lw=1.8,ls=ls,label=name)
   for z in parts:draw(z.exterior.coords,'#0072B2','Whole named road A')
   draw(tp,'#D55E00','Whole target polygon');ax.scatter(*gt,c='#D55E00',marker='x',s=65,label='GT: offline check only')
   if f and f.get('points_xy'):
    xy=np.array(f['points_xy']);d=np.array(f['distance_from_source_m']);order=np.argsort(d);ax.plot(*xy[order].T,color='#009E73',lw=1.5,label='Frozen v4 centerline')
    if p['state']=='unique_field':
     ix=order[::max(1,len(order)//16)];h=np.deg2rad(np.array(f['heading_deg'])[ix]);ax.quiver(xy[ix,0],xy[ix,1],5*np.cos(h),5*np.sin(h),angles='xy',scale_units='xy',scale=1,color='#009E73',width=.006,label='Independent positive travel')
    b=region(x,f['source_ids'])
    if b is not None:
     for z in (list(b.geoms) if b.geom_type=='MultiPolygon' else [b]):draw(z.exterior.coords,'#CC79A7','ALL named source B segments','--')
    ff=FF[f['key']]
    if j==1 and ff['state']=='unique_field':xx=np.array(ff['points_xy']);ax.plot(*xx[order].T,c='#E69F00',ls=':',lw=1.8,label='F: exploratory smoothing')
   ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World X (m)',ylabel='World Y (m)',title='Public map context' if j==0 else 'Offline contour check; geometry not fit to GT');ax.set_aspect('equal');ax.grid(alpha=.18);handles,labels=ax.get_legend_handles_labels();dedup=dict(zip(labels,handles));ax.legend(dedup.values(),dedup.keys(),fontsize=6,loc='best')
  fig.suptitle(f"{label} | {p['state']} | {q['result']} | relation={q['word']}");caption='\n'.join(textwrap.wrap(x['instruction'],145))+f"\n{rid} | {q['reason']}\nActual TRAIN orthomap; image right=+X, up=+Y, no true-north claim. Target GT is used only for evaluation/illustration.\nHistorical/development and new retrospective cases are labeled separately; no orientation is reversed to fit target."
  fig.text(.055,.035,caption,fontsize=9,va='bottom');name=f'case_{k:02d}_{label}';save(fig,name);index.append({'occurrence_id':rid,'figure':name+'.png','kind':label,'instruction':x['instruction'],'result':q['result'],'state':p['state'],'source_ids':f.get('source_ids') if f else [],'reference_ids':x['reference_ids'],'illustration_selection_only':True,'historical_development_case':rid not in {p['occurrence_id'] for p in readlines(RESULT/'new_frames.jsonl.gz')}})
 writej(RESULT/'case_index.json',index);print('PNG/PDF pairs',len(list(FIG.glob('*.png'))))
if __name__=='__main__':run()
