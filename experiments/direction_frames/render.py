"""All figures are derived from real TRAIN records. No generated illustrative data."""
import argparse,json,textwrap,collections
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from analyze import WORDS,OFFSETS,distance,dump,polygon
Image.MAX_IMAGE_PIXELS=None
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':320,'pdf.fonttype':42})
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);args=ap.parse_args();base=Path(__file__).parent;p=base/'results';figdir=base/'figures';figdir.mkdir(exist_ok=True)
 load=lambda name:json.loads((p/name).read_text());R=load('occurrences.json');v=pd.DataFrame([r for r in R if r['status']=='valid']);H=load('hypotheses.json');W={h['word']:h for h in H if h['hypothesis']=='A_world' and h['n']}
 def save(fig,name):
  fig.savefig(figdir/(name+'.png'),dpi=320,bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(4,4,figsize=(16,12),layout='constrained')
 for ax,w in zip(axes.flat,WORDS):
  x=v[v.semantic_word==w].angle_world
  ax.hist(x,bins=np.arange(0,361,15),color=BLUE,edgecolor='white');ax.set(title=f'{w} (n={len(x)})',xlim=(0,360),xticks=[0,90,180,270,360],xlabel='World XY angle (deg)',ylabel='Occurrences')
  if len(x):ax.axvline(W[w]['axis_deg'],color=ORANGE,linestyle='--',label='best cardinal axis (fit)')
 for ax in axes.flat[len(WORDS):]:ax.axis('off')
 fig.suptitle('Original CityNav TRAIN | directly bound, disjoint polygons | 0 deg = +X, CCW');save(fig,'world_histograms')
 fig,axs=plt.subplots(1,4,figsize=(16,4.5),subplot_kw={'projection':'polar'},layout='constrained')
 for ax,w in zip(axs,['left','right','front','behind']):
  a=v[v.semantic_word==w].angle_world.to_numpy();bins=np.linspace(0,2*np.pi,25);n,_=np.histogram(np.deg2rad(a),bins)
  ax.bar(bins[:-1],n,width=np.diff(bins),align='edge',color=BLUE,alpha=.8,edgecolor='white');ax.set_title(f'{w} | n={len(a)}',pad=20);ax.set_xticks(np.deg2rad([0,90,180,270]),['+X','+Y','-X','-Y']);ax.set_theta_zero_location('E');ax.set_theta_direction(1)
 save(fig,'polar_main')
 fig,axs=plt.subplots(3,4,figsize=(16,12),subplot_kw={'projection':'polar'},layout='constrained')
 for j,(frame,col) in enumerate([('World','angle_world'),('Initial yaw','angle_yaw'),('Start to landmark LOS','angle_los')]):
  for i,w in enumerate(['left','right','front','behind']):
   ax=axs[j,i];a=v[v.semantic_word==w][col].to_numpy();bins=np.linspace(0,2*np.pi,17);n,_=np.histogram(np.deg2rad(a),bins);ax.bar(bins[:-1],n,width=np.diff(bins),align='edge',color=[BLUE,ORANGE,GREEN][j],alpha=.8);ax.set_title(f'{frame}: {w} (n={len(a)})',pad=17)
 save(fig,'polar_frames')
 g=load('groups.json');maps=sorted(v['map'].unique());fig,axs=plt.subplots(1,4,figsize=(14,10),layout='constrained')
 for ax,w in zip(axs,['left','right','front','behind']):
  gs={x['group']:x for x in g if x['group_type']=='map' and x['word']==w}
  mat=np.array([gs[m]['consistency'] if m in gs else np.nan for m in maps])[:,None]
  im=ax.imshow(mat,aspect='auto',vmin=0,vmax=1,cmap='cividis');ax.set(xticks=[0],xticklabels=[w],yticks=np.arange(len(maps)),yticklabels=maps if w=='left' else ['']*len(maps))
  for k,m in enumerate(maps):
   if m in gs:ax.text(0,k,f"{gs[m]['consistency']:.0%} | n={gs[m]['n']}",ha='center',va='center',color='white' if gs[m]['consistency']<.55 else 'black',fontsize=8)
 fig.colorbar(im,ax=axs,label='Consistency at global fitted word axis');save(fig,'map_consistency')
 for key in ['target_category','ref_category']:
  cats=sorted({x['group'] for x in g if x['group_type']==key});fig,axs=plt.subplots(1,4,figsize=(15,4),layout='constrained')
  for ax,w in zip(axs,['left','right','front','behind']):
   gs={x['group']:x for x in g if x['group_type']==key and x['word']==w}
   for k,c in enumerate(cats):
    if c not in gs:continue
    z=gs[c];lo,hi=z['cluster_bootstrap95'];ax.errorbar(k,z['consistency'],yerr=[[max(0,z['consistency']-lo)],[max(0,hi-z['consistency'])]],fmt='o',color=BLUE,capsize=3);ax.annotate(f"n={z['n']}",(k,z['consistency']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
   ax.set(title=w,xticks=range(len(cats)),xticklabels=cats,ylim=(-.05,1.12),ylabel='Consistency | cluster 95% CI');ax.tick_params(axis='x',rotation=35)
  save(fig,key+'_consistency')
 matched=load('matched_hypotheses.json');fig,ax=plt.subplots(figsize=(10,5),layout='constrained');ws=['left','right','front','behind']
 for j,(h,label,c) in enumerate([('A_world_crossfit','A: crossfit XY',BLUE),('B_initial_yaw','B: initial yaw',ORANGE),('C_start_landmark_LOS','C: LOS',GREEN)]):
  for i,w in enumerate(ws):
   z=next(s for s in matched if s['word']==w and s['hypothesis']==h);y=z['consistency'];lo,hi=z['cluster95'];ax.errorbar(i+(j-1)*.18,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=4,color=c,label=label if i==0 else None)
 ax.axhline(.25,color='gray',linestyle=':',label='uniform-angle sector chance (25%)');ax.set(xticks=range(4),xticklabels=ws,ylim=(0,1),ylabel='Consistency (90-degree sector) | cluster 95% CI',title='Matched TRAIN support | D unavailable: no directed heading');ax.legend();save(fig,'hypothesis_comparison')
 bs=pd.read_csv(p/'polygon_summary.csv');fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
 for ax,h in zip(axs,['A_world','B_initial_yaw','C_start_landmark_LOS']):
  bottom=np.zeros(4)
  for cl,c in [('consistent',GREEN),('ambiguous','#999999'),('opposite',ORANGE)]:
   vals=np.array([bs[(bs.word==w)&(bs.hypothesis==h)&(bs.tolerance_m==1)&(bs.boundary_class==cl)].n.sum() for w in ws]);ax.bar(ws,vals,bottom=bottom,color=c,label=cl);bottom+=vals
  ax.set(title=h,ylabel='Entire-polygon projected separation (n)');ax.legend(fontsize=8)
 save(fig,'polygon_boundary_checks')
 # Select real examples with full geographic context. Choice is descriptive and exported.
 O=json.loads((args.data_root/'cityrefer_original/objects.json').read_text());E=json.loads((args.data_root/'original_citynav/citynav_train_seen.json').read_text());byid={r['occurrence_id']:r for r in R};cases=[]
 contra=load('same_pair_opposite_descriptions.json')
 for c in contra:
  r=byid[c['first']];cases.append((r,'opposite_descriptions',c['instruction2']))
 # Find strongest complete-boundary matches and opposite examples, do not fabricate.
 b=pd.read_csv(p/'polygon_checks.csv');b=b[(b.tolerance_m==1)&(b.hypothesis=='A_world')]
 used={r['occurrence_id'] for r,_,_ in cases}
 for w in ['left','right','front','behind','above','below']:
  z=b[(b.word==w)&(b.boundary_class=='consistent')].sort_values('projection_gap_m',ascending=False)
  if len(z):
   rid=z.iloc[0].occurrence_id
   if rid not in used:cases.append((byid[rid],'fits_fitted_world_axis',''));used.add(rid)
 for reason in ['ambiguous_overlap_or_near_boundary','indirect_reference_np','duplicate_landmark_name']:
  z=[r for r in R if r['reason']==reason and r['word'] in ['left','right','front','behind','above','below']]
  if z:cases.append((z[0],'unresolved_'+reason,''))
 case_index=[]
 for k,(r,kind,extra) in enumerate(cases):
  ep=E[r['episode']];m=r['map'];t=O[m][r['target_id']];gt=ep['target_positions'][0];ref=O[m].get(r.get('ref_id')) if r.get('ref_id') else None
  # For unbound examples, show named context only, with no invented relation arrow.
  contexts=[]
  if ref is None:
   from analyze import name_spans
   for hit in name_spans(r['instruction'],O[m]):
    contexts.extend(O[m][i] for i in hit['ids'][:3])
  polys=[np.array(t['contour'])]+([np.array(ref['contour'])] if ref else [np.array(o['contour']) for o in contexts if o.get('contour')])
  points=np.vstack([x for x in polys if len(x)]);mn=points.min(0);mx=points.max(0);pad=max(12,float(max(mx-mn)*.15));mn-=pad;mx+=pad
  fig=plt.figure(figsize=(10,10),layout='constrained');gs=fig.add_gridspec(2,1,height_ratios=[4,1.5]);ax=fig.add_subplot(gs[0]);txt=fig.add_subplot(gs[1]);txt.axis('off')
  with rasterio.open(args.data_root/'rgbd'/f'{m}.tif') as src:
   tr=src.transform;ci0=max(0,int(np.floor((mn[0]-tr.c)/tr.a)));ci1=min(src.width,int(np.ceil((mx[0]-tr.c)/tr.a)));ri0=max(0,int(np.floor((mx[1]-tr.f)/tr.e)));ri1=min(src.height,int(np.ceil((mn[1]-tr.f)/tr.e)))
   if ci1>ci0 and ri1>ri0:
    with Image.open(args.data_root/'rgbd'/f'{m}.png') as img:patch=np.array(img.crop((ci0,ri0,ci1,ri1)))
    ax.imshow(patch,extent=[tr.c+ci0*tr.a,tr.c+ci1*tr.a,tr.f+ri1*tr.e,tr.f+ri0*tr.e],origin='upper',alpha=.8)
  def draw(o,color,label,ls='-'):
   xy=np.array(o['contour']);xy=np.vstack([xy,xy[0]]);ax.plot(xy[:,0],xy[:,1],color=color,linewidth=2,linestyle=ls,label=label);ax.fill(xy[:,0],xy[:,1],color=color,alpha=.13)
  draw(t,ORANGE,'Target contour');ax.scatter(*gt[:2],s=50,color=ORANGE,marker='x',label='Target GT')
  angle_note='No target-reference angle assigned'
  if ref:
   draw(ref,BLUE,'Reference contour');rx,ry=ref['position'][:2];ax.scatter(rx,ry,color=BLUE,s=22);ax.annotate('',xy=gt[:2],xytext=(rx,ry),arrowprops={'arrowstyle':'->','color':'black','lw':1.6});angle_note=f"Reference -> target: theta={r.get('angle_world',float('nan')):.2f} deg; dx={gt[0]-rx:.2f}m, dy={gt[1]-ry:.2f}m"
   ax.text((rx+gt[0])/2,(ry+gt[1])/2,f"{r.get('angle_world',float('nan')):.1f} deg",bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'},fontsize=9)
  for o in contexts:draw(o,'#777777','Named context ONLY (unbound)','--')
  ax.set(xlim=(mn[0],mx[0]),ylim=(mn[1],mx[1]),xlabel='World X (m) | image right = +X',ylabel='World Y (m) | image up = +Y',title=f"{kind}\n{m} | target {r['target_id']} | {r['word']}");ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
  info=r['instruction']+('\nOpposite wording for SAME pair: '+extra if extra else '')+'\n'+angle_note+'\n'+r['occurrence_id']+' | Geographic north uncalibrated; axes are world XY.'
  txt.text(0,1,'\n'.join(textwrap.fill(line,105) for line in info.split('\n')),va='top',fontsize=9)
  name=f'case_{k:02d}_{kind}';save(fig,name);case_index.append({'figure':name+'.png','occurrence_id':r['occurrence_id'],'kind':kind,'instruction':r['instruction'],'opposite_instruction':extra,'angle_annotation':angle_note})
 dump(p/'case_index.json',case_index)
 print('Rendered',len(list(figdir.glob('*.png'))),'PNG figures + PDF versions')
if __name__=='__main__':main()
