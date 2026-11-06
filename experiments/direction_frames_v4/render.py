"""Real-statistic plots and true TRAIN orthomap cases (no generated imagery)."""
from shared import *
from infer import region,anchor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from PIL import Image
import rasterio,textwrap
from shapely.geometry import Polygon
Image.MAX_IMAGE_PIXELS=None
FIG=BASE4/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','savefig.dpi':320,'pdf.fonttype':42,'font.size':10})
def save(fig,name):
 fig.savefig(FIG/(name+'.png'),dpi=320,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def run():
 S=readj(OUT/'summary.json');Q=readlines(OUT/'geometry_checks.jsonl.gz');X={r['occurrence_id']:r for r in readlines(OUT/'public_inputs.jsonl.gz')};P={r['occurrence_id']:r for r in readlines(OUT/'frozen_frames.jsonl.gz')};G={r['occurrence_id']:r for r in readlines(OUT/'evaluation_gt.jsonl.gz')};fields={r['key']:r for r in readlines(OUT/'road_fields.jsonl.gz')};byQ={r['occurrence_id']:r for r in Q}
 fig,ax=plt.subplots(figsize=(10,5),layout='constrained');cohorts=['core_all_named','core_bound','road_semantic','new_semantic_road_anchors'];bottom=np.zeros(len(cohorts))
 for state,label,color in [('unique','Unique directed FIELD','#009E73'),('multiple','Multiple / unsigned','#E69F00'),('unknown','Unknown','#999999')]:
  vals=[]
  for c in cohorts:
   s=next(s for s in S if s['cohort']==c and s['tier']=='all');vals.append(s[state]/s['n'])
  ax.barh(cohorts,vals,left=bottom,color=color,label=label);bottom+=vals
 ax.set(xlim=(0,1),xlabel='Fraction before target GT evaluation',title='One heading is never forced; road fields are spatially varying');ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3);save(fig,'identifiability_coverage')
 fig,ax=plt.subplots(figsize=(9,5),layout='constrained');labels=['All unique fields','High confidence','New road-anchor syntax'];rr=[[r for r in Q if r['state']=='unique_field'],[r for r in Q if r['state']=='unique_field' and r['parse_tier']=='high'],[r for r in Q if r['state']=='unique_field' and r['origin']=='v4_explicit_road_anchor']];bot=np.zeros(3)
 for result,color in [('consistent','#009E73'),('conflict','#D55E00'),('ambiguous','#999999')]:
  vals=[sum(r['result']==result for r in q) for q in rr];ax.bar(labels,vals,bottom=bot,color=color,label=result)
  for i,v in enumerate(vals):
   if v:ax.text(i,bot[i]+v/2,str(v),ha='center',va='center',color='white')
  bot+=vals
 ax.set(ylabel='Relation count; no ambiguous samples hidden',title='Full-contour road-side evaluation on phase-1 unique support');ax.legend();save(fig,'road_evaluation_denominators')
 M=readj(OUT/'leave_map_out.json');M=[m for m in M if m['unique']];fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
 for i,m in enumerate(M):
  if m['conditional_consistency'] is None:continue
  y=m['conditional_consistency'];lo,hi=m['wilson95'];ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=4,color='#0072B2');ax.text(i,.05,f"{m['consistent']}/{m['geometry_scorable']}\nunique {m['unique']}",ha='center',fontsize=8)
 ax.set(xticks=range(len(M)),xticklabels=[m['heldout_map'].replace('_block_','\n') for m in M],ylim=(0,1.06),ylabel='Conditional consistency (Wilson 95% CI)',title='Same frozen rules on each held-out map; no fitting, not blind external validation');save(fig,'heldout_maps')
 B=readj(OUT/'matched_baseline_summary.json');fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
 for i,b in enumerate(B):
  y=b['paired_difference'];lo,hi=b['object_cluster95'];ax.errorbar(i,100*y,yerr=[[100*(y-lo)],[100*(hi-y)]],fmt='o',capsize=5,color='#0072B2')
 ax.axhline(0,color='gray',ls=':');ax.set(xticks=range(len(B)),xticklabels=[b['baseline'].replace('_','\n') for b in B],ylabel='Paired confirmed-fraction difference (pp)',title='Same 44 phase-1 unique relations; ambiguous = unconfirmed');save(fig,'matched_baselines')
 N=readj(OUT/'negative_controls.json');fig,ax=plt.subplots(figsize=(9,5),layout='constrained');names=[];vals=[]
 for n in N:
  names.append(n['control'].replace('_','\n'));vals.append(n.get('null_mean',n.get('confirmed_fraction_all_support',0)))
 ax.bar(range(len(vals)),vals,color='#999999');observed=N[0]['observed_confirmed_fraction'];ax.axhline(observed,color='#009E73',label=f'observed full support {observed:.1%}');ax.set(xticks=range(len(names)),xticklabels=names,ylim=(0,1),ylabel='Confirmed fraction on original unique support',title='Abstentions and ambiguous contours retained in denominator');ax.legend();save(fig,'negative_controls')
 fig,ax=plt.subplots(figsize=(8,5),layout='constrained');ranges=[p['heading_support']['width_deg'] for p in P.values() if p.get('heading_support')];ax.hist(ranges,bins=np.arange(0,361,20),color='#E69F00',edgecolor='white');ax.axvline(20,color='black',ls=':',label='20deg declared narrow-support cutoff');ax.set(xlabel='Admissible observer-heading range width (deg)',ylabel='Observed TRAIN relations',title='Named regions do not identify exact viewpoint or aim point');ax.legend();save(fig,'observer_region_ambiguity')
 chosen=[('cambridge_block_28:8:2:142','known_boathouse_range'),('birmingham_block_3:19:2:128','new_Bragg_road_entry'),('cambridge_block_4:14:4:92','new_St_Peters_entry'),('cambridge_block_28:84:0:91','new_Aylestone_entry'),('cambridge_block_28:17:3:26','Abbey_conflict'),('birmingham_block_1:130:4:109','car_contour_ambiguous'),('cambridge_block_13:258:1:183','facing_without_origin'),('birmingham_block_10:46:2:22','intrinsic_unsigned')];index=[]
 for k,(rid,label) in enumerate(chosen):
  x=X[rid];p=P[rid];g=G[rid];q=byQ[rid];rp=region(x,x['reference_ids']);tp=np.array(g['target_polygon']);gt=np.array(g['subject_xy']);rxy=np.array(g['reference_xy']);f=fields.get(p.get('road_field_key'));refpts=np.concatenate([np.array(z.exterior.coords) for z in (list(rp.geoms) if rp.geom_type=='MultiPolygon' else [rp])]);allpts=np.vstack([refpts,tp]);mn=allpts.min(0)-15;mx=allpts.max(0)+15
  # Independent full-road context and a GT-centered display crop are both shown;
  # crop selection never enters frozen inference or numerical statistics.
  panels=[(mn,mx),(np.minimum(tp.min(0),gt)-22,np.maximum(tp.max(0),gt)+22)];fig,axs=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(bottom=.25,top=.91,wspace=.2)
  with rasterio.open(DATA/'rgbd'/f"{x['map']}.tif") as src:tr=src.transform;width=src.width;height=src.height
  for j,(ax,(lo,hi)) in enumerate(zip(axs,panels)):
   c0=max(0,int((lo[0]-tr.c)/tr.a));c1=min(width,int(np.ceil((hi[0]-tr.c)/tr.a)));r0=max(0,int((hi[1]-tr.f)/tr.e));r1=min(height,int(np.ceil((lo[1]-tr.f)/tr.e)))
   with Image.open(DATA/'rgbd'/f"{x['map']}.png") as im:img=np.array(im.crop((c0,r0,c1,r1)))
   extent=[tr.c+c0*tr.a,tr.c+c1*tr.a,tr.f+r1*tr.e,tr.f+r0*tr.e];ax.imshow(img,extent=extent,origin='upper')
   def contour(points,color,label,ls='-'):
    z=np.array(points);ax.plot(*np.vstack([z,z[0]]).T,c=color,lw=1.8,ls=ls,label=label)
   for z in (list(rp.geoms) if rp.geom_type=='MultiPolygon' else [rp]):contour(z.exterior.coords,'#0072B2','Named reference Polygon')
   contour(tp,'#D55E00','Relation subject Polygon');ax.scatter(*gt,marker='x',s=75,c='#D55E00',label='Target GT (offline only)')
   if f and f.get('points_xy'):
    xy=np.array(f['points_xy']);heads=np.array(f['heading_deg']);dist=np.array(f['distance_from_source_m']);order=np.argsort(dist);ax.plot(*xy[order].T,color='#009E73',lw=1.5,label='Frozen directed centerline');ix=order[::max(1,len(order)//16)];u=np.stack([np.cos(np.deg2rad(heads[ix])),np.sin(np.deg2rad(heads[ix]))],axis=1);ax.quiver(xy[ix,0],xy[ix,1],u[:,0]*5,u[:,1]*5,color='#009E73',angles='xy',scale_units='xy',scale=1,width=.005,label='Positive travel direction')
    src=region(x,f['source_ids'])
    if src is not None:
     for z in (list(src.geoms) if src.geom_type=='MultiPolygon' else [src]):contour(z.exterior.coords,'#CC79A7','Independent source road','--')
    if j==1:
     # Decision regions from the FROZEN field, evaluated on map grid without GT.
     gx,gy=np.meshgrid(np.linspace(lo[0],hi[0],100),np.linspace(lo[1],hi[1],100));points=np.stack([gx.ravel(),gy.ravel()],axis=1);near=np.argmin(np.sum((points[:,None,:]-xy[None,:,:])**2,axis=2),axis=1);h=np.deg2rad(heads[near]);n=np.stack([-np.sin(h),np.cos(h)],axis=1);side=((points-xy[near])*n).sum(1).reshape(gx.shape);ax.contourf(gx,gy,side,levels=[-1e9,-1,1,1e9],colors=['#56B4E9','#FFFFFF','#E69F00'],alpha=.16)
     ax.text(.02,.96,'Orange: LEFT | Blue: RIGHT\nGray band: 1m boundary tolerance',transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
   elif p.get('heading_support'):
    support=p['heading_support'];center=np.array(p['origin_xy']);rad=25.;ax.add_patch(Wedge(center,rad,support['start_deg'],support['start_deg']+support['width_deg'],color='#E69F00',alpha=.18,label='Full admissible observer range'))
    for cc in p['candidates']:
     if cc['type']=='observer_region_direction_range':
      for oid in cc['destination_ids']:
       if oid in x['scene']:contour(x['scene'][oid]['contour'],'#009E73','Facing destination region','--')
   else:
    for c in p.get('candidates',[]):
     if c['type']=='building_PCA_axis':
      h=np.deg2rad(c['heading_deg']);ax.arrow(rxy[0],rxy[1],15*np.cos(h),15*np.sin(h),color='#777777',width=.25,head_width=2,length_includes_head=True)
   ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World X (m)',ylabel='World Y (m)',title='Independent map/reference context' if j==0 else 'Geometry check zoom; no geographic-north claim');ax.set_aspect('equal');ax.grid(alpha=.18)
   handles,labels=ax.get_legend_handles_labels();dedup=dict(zip(labels,handles));ax.legend(dedup.values(),dedup.keys(),fontsize=6,loc='lower right')
  fig.suptitle(f"{label} | {p['frame_type']} / {p['state']} | {q['result']}");caption='\n'.join(textwrap.wrap(x['instruction'],150))+f"\n{rid} | word={x['word']} | {q['reason']}\n{p['reason']}\nOriginal TRAIN orthomap. Heading inferred before GT. Road field is a geometry proxy, not verified author yaw."
  fig.text(.055,.045,caption,fontsize=9,va='bottom');name=f'case_{k:02d}_{label}';save(fig,name);index.append({'occurrence_id':rid,'figure':name+'.png','instruction':x['instruction'],'frame_type':p['frame_type'],'state':p['state'],'geometry_result':q['result'],'reason':q['reason'],'reference_ids':x['reference_ids'],'source_evidence':p['evidence'],'case_selection_for_illustration_only':True,'known_development_example':label=='known_boathouse_range'})
 writej(OUT/'case_index.json',index);print('Figures',len(list(FIG.glob('*.png'))))
if __name__=='__main__':run()
