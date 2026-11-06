"""Statistical figures and real-map case re-audits; no synthetic visual outcomes."""
import argparse,gzip,json,textwrap
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from PIL import Image
import rasterio
from analyze_frames import cluster_ci,MAIN
from geometry_frames import poly,vertices
from extract import dump
BASE=Path(__file__).parent;P=BASE/'results';F=BASE/'figures';F.mkdir(exist_ok=True)
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';Image.MAX_IMAGE_PIXELS=None
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':320,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=320,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def run(root):
 load=lambda n:json.loads((P/n).read_text());C=load('corpus.json');D=pd.read_csv(P/'frame_evaluations.csv');Q=pd.read_csv(P/'crossfit_predictions.csv');cov=load('coverage.json');mats=load('map_transformations.json');M=load('matched_comparisons.json');pairs=[json.loads(x) for x in gzip.open(P/'geometry_pairs.jsonl.gz','rt')]
 fig,ax=plt.subplots(figsize=(11,5),layout='constrained');ws=[w for w in ['left','right','front','behind','back','above','below','north','south'] if any(x['word']==w for x in cov)];Z={x['word']:x for x in cov};xs=np.arange(len(ws))
 for j,(k,label,c) in enumerate([('occurrences','Named-landmark relation candidates','#999999'),('text_bound','Independently bound subject + landmark',BLUE),('disjoint_all','Full effective: disjoint contours',GREEN),('disjoint_high','High confidence: disjoint contours',ORANGE)]):ax.bar(xs+(j-1.5)*.2,[Z[w][k] for w in ws],width=.2,label=label,color=c)
 ax.set(xticks=xs,xticklabels=ws,ylabel='Named-landmark relation occurrences',title='Population excludes target-to-anonymous-neighbor relations');ax.legend(fontsize=8);save(fig,'landmark_coverage')
 fig,ax=plt.subplots(figsize=(13,5),layout='constrained');names=mats['unique_candidate_names']
 for j,subset in enumerate(['high','all_effective']):
  for i,h in enumerate(names):
   z=D[(D.geometry_status=='disjoint')&D.word.isin(MAIN)&(D.hypothesis==h)]
   if subset=='high':z=z[z.parse_tier=='high']
   y=z.sector_correct.mean();lo,hi=cluster_ci(z,z.sector_correct.astype(float));ax.errorbar(i+(j-.5)*.15,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=3,color=[ORANGE,BLUE][j],label=subset if i==0 else None)
 ax.set(xticks=range(len(names)),xticklabels=[x.replace('A_image_','') for x in names],ylim=(0,1),ylabel='Joint directional dictionary consistency | cluster 95% CI',title='8 unique rotations/reflections (12 named tests, 4 duplicates) | descriptive, not crossfit');ax.tick_params(axis='x',rotation=25);ax.legend();save(fig,'joint_map_transformations')
 fig,ax=plt.subplots(figsize=(11,6),layout='constrained');hs=[h for h in ['B_start_to_reference','B_reference_to_start','C_initial_yaw','C_initial_actual_motion','C_motion_approaching_reference','C_motion_approaching_target'] if any(z['hypothesis']==h for z in M)]
 for j,subset in enumerate(['high','all_effective']):
  for i,h in enumerate(hs):
   z=next(x for x in M if x['subset']==subset and x['hypothesis']==h);y=z['delta_consistency'];lo,hi=z['delta_cluster95'];ax.errorbar(y,i+(j-.5)*.18,xerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=4,color=[ORANGE,BLUE][j],label=subset if i==0 else None);ax.annotate(f"n={z['n']}",(y,i+(j-.5)*.18),xytext=(5,3),textcoords='offset points',fontsize=8)
 ax.axvline(0,linestyle=':',color='black');ax.set(yticks=range(len(hs)),yticklabels=hs,xlabel='Paired consistency difference vs joint-D4 crossfit on SAME support',title='Motion involving future trajectory is offline diagnosis only');ax.legend();save(fig,'matched_reference_comparison')
 fig,axs=plt.subplots(1,4,figsize=(16,4.5),subplot_kw={'projection':'polar'},layout='constrained')
 for ax,w in zip(axs,['left','right','front','behind']):
  for subset,c in [('all_effective',BLUE),('high',ORANGE)]:
   r=[p for p in pairs if p['semantic_word']==w and p['geometry_status']=='disjoint' and (subset!='high' or p['confidence_tier']=='high')];a=np.deg2rad([p['angle_world'] for p in r]);bins=np.linspace(0,2*np.pi,25);n,_=np.histogram(a,bins);ax.stairs(n/max(1,len(a)),bins,color=c,label=f'{subset}: n={len(a)}',linewidth=2)
  ax.set_title(w,pad=20);ax.set_xticks(np.deg2rad([0,90,180,270]),['+X','+Y','-X','-Y']);ax.legend(loc='upper right',bbox_to_anchor=(1.15,-.1),fontsize=8)
 save(fig,'landmark_angles_polar')
 # Crossfit subgroup evidence, including view/orientation cue strata.
 sg=[]
 for (sub,scheme),q in Q.groupby(['subset','scheme']):
  for key in ['map','explicit_view','target_category']:
   for val,g in q.groupby(key):sg.append({'subset':sub,'scheme':scheme,'group_type':key,'group':str(val),'n':len(g),'consistency':float(g.sector_correct.mean()),'cluster95':cluster_ci(g,g.sector_correct.astype(float))})
 dump(P/'crossfit_strata.json',sg);pd.DataFrame(sg).to_csv(P/'crossfit_strata.csv',index=False)
 fig,ax=plt.subplots(figsize=(12,5),layout='constrained');data=[g for g in sg if g['subset']=='all_effective' and g['scheme']=='target_5fold' and g['group_type']=='map'];data=sorted(data,key=lambda g:g['group'])
 for i,z in enumerate(data):
  y=z['consistency'];lo,hi=z['cluster95'];ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=3,color=BLUE);ax.annotate(f"n={z['n']}",(i,y),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
 ax.set(xticks=range(len(data)),xticklabels=[z['group'].replace('_block_',' b') for z in data],ylim=(-.05,1.1),ylabel='Joint-D4 object crossfit | cluster 95% CI',title='Different map distributions do not establish different map conventions');ax.tick_params(axis='x',rotation=55);save(fig,'crossfit_by_map')
 A=pd.read_csv(P/'unsigned_axis_compatibility.csv');fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for ax,cat in zip(axs,['Building','TrafficRoad']):
  ws2=['left','right','front','behind'];vals=[]
  for i,w in enumerate(ws2):
   g=A[(A.category==cat)&(A.word==w)];vals.append(len(g));
   if len(g):
    y=g.axis_compatible.mean();lo,hi=cluster_ci(g,g.axis_compatible.astype(float));ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',color=BLUE,capsize=4);ax.annotate(f'n={len(g)}',(i,y),xytext=(0,8),textcoords='offset points',ha='center')
  ax.axhline(.5,color='gray',linestyle=':');ax.set(title=cat+' | UNSIGNED AXIS ONLY',xticks=range(4),xticklabels=ws2,ylim=(0,1.12),ylabel='Axis compatibility modulo 180 deg (not signed accuracy)')
 save(fig,'unsigned_axes_not_directions')
 # Four real-map case panels: all show exactly the same actual imagery and contours.
 cases=load('four_case_reaudit.json');O=json.loads((root/'cityrefer_original/objects.json').read_text());index=[]
 for c in cases:
  m=c['map'];t=c['target_object'];ref=c['reference_object'];rxy=np.array(ref['position'][:2]);polys=[np.array(t['contour']),np.array(ref['contour'])];extra=c.get('independent_river_facing');pts=np.vstack(polys)
  if extra:pts=np.vstack([pts,extra['water_boundary_point']])
  mn=pts.min(0)-14;mx=pts.max(0)+14
  with rasterio.open(root/'rgbd'/f'{m}.tif') as src:
   tr=src.transform;c0=max(0,int(np.floor((mn[0]-tr.c)/tr.a)));c1=min(src.width,int(np.ceil((mx[0]-tr.c)/tr.a)));r0=max(0,int(np.floor((mx[1]-tr.f)/tr.e)));r1=min(src.height,int(np.ceil((mn[1]-tr.f)/tr.e)))
   with Image.open(root/'rgbd'/f'{m}.png') as img:patch=np.array(img.crop((c0,r0,c1,r1)))
   ext=[tr.c+c0*tr.a,tr.c+c1*tr.a,tr.f+r1*tr.e,tr.f+r0*tr.e]
  ann=next(a for a in c['source_annotations_TRAIN_only'] if a['instruction']==c['instruction1']);yaw=ann['initial_yaw_deg'];axis=c['reference_unsigned_axis'];third=extra['heading_deg'] if extra else axis['axis_deg'] if axis else 0
  frames=[(0,'World XY | +Y = image up'),(yaw,'Initial UAV yaw (proxy, not author yaw)'),(third,'Text: face unique stream (proxy)' if extra else 'PCA axis sign 0 | +/- unresolved')]
  fig=plt.figure(figsize=(17,10),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[4,1.35])
  for j,(heading,title) in enumerate(frames):
   ax=fig.add_subplot(grid[0,j]);display_rotation=0 if j==0 else 90-heading;a=np.deg2rad(display_rotation);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);convert=lambda x:(np.array(x)-rxy)@rot.T
   trans=Affine2D().translate(-rxy[0],-rxy[1]).rotate_deg(display_rotation)+ax.transData;ax.imshow(patch,extent=ext,origin='upper',transform=trans)
   def draw(o,color,label,ls='-'):
    xy=convert(o['contour']);xy=np.vstack([xy,xy[0]]);ax.plot(xy[:,0],xy[:,1],color=color,linewidth=2,linestyle=ls,label=label)
   draw(t,ORANGE,'Relation subject contour');draw(ref,BLUE,'Named landmark contour')
   if extra:draw({'contour':extra['water_polygon']},GREEN,'Unique water contour','--')
   x=convert([t['position'][:2]])[0];ax.scatter(*x,c=ORANGE,marker='x',s=45);ax.annotate('',xy=x,xytext=(0,0),arrowprops={'arrowstyle':'->','lw':1.5});ax.scatter(0,0,c=BLUE,s=20)
   # Neighbor labels are visual context, never used to choose a reference.
   for oid,o in O[m].items():
    if oid in {c['target_id'],c['ref_id']} or o['object_type'] not in {'Building','Car'}:continue
    pos=np.array(o['position'][:2]);
    if np.all(pos>=mn) and np.all(pos<=mx):
     q=convert([pos])[0];ax.text(*q,str(oid),color='#444444',fontsize=6,bbox={'facecolor':'white','alpha':.5,'edgecolor':'none','pad':.3})
   limits=convert(np.array([[ext[0],ext[2]],[ext[0],ext[3]],[ext[1],ext[2]],[ext[1],ext[3]]]))
   ax.set(xlim=(limits[:,0].min(),limits[:,0].max()),ylim=(limits[:,1].min(),limits[:,1].max()),xlabel='X-offset (m)' if j==0 else 'Right offset (m)',ylabel='Y-offset (m)' if j==0 else 'Forward offset (m)',title=title);ax.set_aspect('equal');ax.grid(alpha=.2)
   if j==0:
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x+rxy[0]:.0f}'));ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y+rxy[1]:.0f}'));ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
   if j==2 and not extra:
    extent=max(8,min(20,float(max(mx-mn)/4)));ax.annotate('',xy=(0,extent),xytext=(0,-extent),arrowprops={'arrowstyle':'<->','color':'black','lw':2})
   ax.legend(fontsize=6,loc='best')
  txt=fig.add_subplot(grid[1,:]);txt.axis('off');body='1. '+c['instruction1']+'\n2. '+c['instruction2']
  if extra:body+='\nOther TRAIN annotation: '+next(a['instruction'] for a in c['source_annotations_TRAIN_only'] if 'stand at' in a['instruction'])+f"\nIndependent stream-facing right error: {extra['right_error_deg']:.2f} deg; uses landmark + water only."
  body+=f"\n{m} | subject={c['target_id']}, landmark={c['ref_id']} | theta={c['world_angle_deg']:.2f} deg | named reference candidates={c['same_named_reference_ids']}\nReprojected real orthomap, not original annotator-camera image. Gray IDs are context only."
  txt.text(0,1,'\n'.join(textwrap.fill(x,165) for x in body.split('\n')),va='top',fontsize=9);fig.suptitle(f"Case {c['case_number']} re-audit: binding confirmed, shared author frame unverified")
  name=f"case_{c['case_number']:02d}_reaudit";save(fig,name);index.append({'case':c['case_number'],'figure':name+'.png','geographic_north_calibrated':False,'original_annotator_camera_recovered':False})
 dump(P/'figure_index.json',index);print('rendered',len(list(F.glob('*.png'))),'PNG/PDF pairs')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True,type=Path);a=ap.parse_args();run(a.data_root)
