"""Draw one measured three-panel teaser. No model or DATA needed for public mode.

python scripts/draw_overview.py --output-dir figures
python scripts/draw_overview.py --include-rgb --output-dir ../local_only/overview
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon,Rectangle
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from shapely.geometry import box
from shapely.ops import unary_union
from PIL import Image
BLUE='#0078ba'; ORANGE='#ed8a20'; RED='#d12d32'; INK='#182b3c'; CYAN='#00b4c5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.45,'text.color':INK,'axes.labelcolor':INK,'svg.hashsalt':'geombelief-real-overview'})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(ax,text,xy,offset=(0,0),color=INK,size=6.5,ha='left'):
 return ax.annotate(text,xy,xytext=offset,textcoords='offset points',ha=ha,va='center',fontsize=size,color=color,zorder=30,bbox=dict(boxstyle='round,pad=.18',fc='white',ec='none',alpha=.94))
def main():
 default_root=Path(__file__).resolve().parents[1]
 if not (default_root/'OVERVIEW_EVIDENCE.json').exists():default_root=default_root/'figures/author'
 ap=argparse.ArgumentParser();ap.add_argument('--source-dir',type=Path,default=default_root);ap.add_argument('--output-dir',type=Path);ap.add_argument('--include-rgb',action='store_true');args=ap.parse_args()
 root=args.source_dir;E=json.loads((root/'OVERVIEW_EVIDENCE.json').read_text());D=np.load(root/'source/overview_fields.npz');M=json.loads((root/'source/map.json').read_text())
 assert sha(root/'source/overview_fields.npz')==E['source_array_sha256']
 out=args.output_dir or root/'figures';out.mkdir(parents=True,exist_ok=True)
 if args.include_rgb and 'local_only' not in out.resolve().parts:raise ValueError('RGB output must be inside local_only')
 xy=D['world_xy'];grid=xy.reshape(28,28,2);gt=D['target_xyz'][:2];poses=D['poses'];top=xy[D['B0'].argmax()];v=E['display_world_bounds'];norm=LogNorm(1e-5,float(max(D['B0'].max(),D['Bt'].max())))
 # Floor affects only colors below 1e-5; the same scale is used for both fields.
 rgb=None
 if args.include_rgb:
  raster=Path(E['raster']['local_path']);assert sha(raster)==E['raster']['sha256'];rgb=np.array(Image.open(raster))
 fig=plt.figure(figsize=(7.2,3.0),facecolor='white')
 fig.text(.018,.966,'“'+E['instruction_verbatim']+'”',fontsize=8.1,fontstyle='italic',va='center')
 fig.text(.018,.906,'(a)  Language-guided target search',fontsize=8.0,fontweight='bold')
 fig.text(.350,.906,'(b)  Map-geometric goal grounding',fontsize=8.0,fontweight='bold')
 fig.text(.682,.906,'(c)  Belief-guided navigation',fontsize=8.0,fontweight='bold')
 axes=[fig.add_axes([x,.205,.300,.66]) for x in [.018,.350,.682]]
 # Equal square data windows use physical aspect, centered in the available band.
 for i,ax in enumerate(axes):
  ax.set(xlim=(v[0],v[2]),ylim=(v[1],v[3]),aspect='equal');ax.set_facecolor('#f1f4f4')
  if rgb is not None:ax.imshow(rgb,extent=E['raster']['extent'],origin='upper',interpolation='nearest',zorder=0)
  else:
   for r in M['context']:
    p=np.array(r['contour'])
    if len(p)>2:ax.add_patch(Polygon(p,facecolor='#dbe3e4',edgecolor='#a9b8bc',lw=.4,zorder=0))
  for r in M['references']:
   p=np.array(r['contour']);ax.add_patch(Polygon(p,facecolor=BLUE,edgecolor=BLUE,alpha=.22,lw=.8,zorder=5))
   ax.plot(*np.vstack([p,p[:1]]).T,color=BLUE,lw=.7,zorder=6)
  ax.set_xticks([]);ax.set_yticks([])
  for spine in ax.spines.values():spine.set_edgecolor('#bac7cb')
  ax.scatter(*gt,marker='*',s=54,c=RED,edgecolor='white',lw=.5,zorder=25)
  # A true metric scale; no north arrow without a verified compass transform.
  ax.plot([572,622],[844,844],color='white' if rgb is not None else INK,lw=1.4,zorder=30)
  label(ax,'50 m',(597,850),size=5.8,ha='center')
  if i:
   im=ax.pcolormesh(grid[...,0],grid[...,1],D['B0' if i==1 else 'Bt'].reshape(28,28),shading='nearest',cmap='magma',norm=norm,alpha=.72,zorder=2,rasterized=False)
   ax.scatter(*top,marker='x',s=42,c=ORANGE,lw=1.5,zorder=26)
 a,b,c=axes
 for j,k in enumerate(E['candidate_ids_for_display']):
  q=xy[k];a.scatter(*q,s=37,facecolors='none',edgecolors=ORANGE,lw=1.45,zorder=24)
  label(a,f'C{j+1}',q,(-9,-12) if j==0 else (-9,9),color=ORANGE,size=6.8,ha='right')
 a.scatter(*poses[0,:2],marker='^',s=42,c=BLUE,edgecolor='white',lw=.6,zorder=26)
 label(a,'UAV start',poses[0,:2],(-5,10),ha='right',color=BLUE)
 label(a,'Aldridge Road',(733,850),(-3,10),color=BLUE,ha='right')
 # Point exactly to the true nearest boundary; geometric angle is measured, not rewritten language.
 bp=np.array(E['geometry']['nearest_road_boundary_xy'])
 b.annotate('',top,xytext=bp,arrowprops=dict(arrowstyle='<->',color=BLUE,lw=1.1,shrinkA=0,shrinkB=0),zorder=28)
 b.annotate('17.7 m to road boundary\n135° from +X (measured)',xy=(top+bp)/2,xytext=(570,958),fontsize=6.4,color=BLUE,va='center',bbox=dict(fc='white',ec='none',alpha=.96,pad=2),arrowprops=dict(arrowstyle='-',color=BLUE,lw=.65,connectionstyle='angle,angleA=0,angleB=90,rad=3'),zorder=29)
 label(b,r'$B_0$',(573,1055),size=10)
 label(b,'Top-1',top,(-12,-8),color=ORANGE,ha='right')
 label(b,'GT',gt,(-13,12),color=RED,ha='right')
 # Exact current supported candidate cells, not the union with history and not visibility.
 valid=D['current_roi_valid'].any(-1);dx=float(np.median(np.diff(np.unique(xy[:,0]))));dy=float(np.median(np.diff(np.unique(xy[:,1]))))
 supported=unary_union([box(q[0]-dx/2,q[1]-dy/2,q[0]+dx/2,q[1]+dy/2) for q in xy[valid]])
 if not supported.is_empty:
  for p in (list(supported.geoms) if hasattr(supported,'geoms') else [supported]):
   c.add_patch(Polygon(np.array(p.exterior.coords),fill=False,edgecolor=CYAN,lw=1.1,hatch='////',zorder=13))
 footprint=D['footprint_world'];c.add_patch(Polygon(footprint,fill=False,edgecolor=CYAN,lw=.8,ls=(0,(3,2)),zorder=12))
 line,=c.plot(poses[:,0],poses[:,1],color=BLUE,lw=1.5,zorder=17);line.set_path_effects([pe.Stroke(linewidth=2.4,foreground='white'),pe.Normal()])
 c.scatter(*poses[0,:2],marker='^',s=31,c=BLUE,edgecolor='white',lw=.5,zorder=26)
 c.scatter(*poses[-1,:2],marker='s',s=55,facecolors='none',edgecolors=BLUE,lw=1,zorder=27)
 c.annotate('Goal / stop',poses[-1,:2],xytext=(653,875),fontsize=6.5,color=BLUE,arrowprops=dict(arrowstyle='-',color=BLUE,lw=.7),bbox=dict(fc='white',ec='none',alpha=.95,pad=1.5),zorder=30)
 c.annotate('Valid ROI support',xy=xy[valid][0],xytext=(567,944),fontsize=6.3,color='#007782',arrowprops=dict(arrowstyle='-',color='#007782',lw=.65),bbox=dict(fc='white',ec='none',alpha=.95,pad=1.5),zorder=30)
 label(c,r'$B_7$',(791,1055),size=10,ha='right')
 if args.include_rgb:
  row=E['observation_crops'][-1];path=Path(row['local_path']);assert sha(path)==row['sha256'];obs=np.load(path)
  ins=c.inset_axes([.045,.50,.34,.34]);ins.imshow(obs['bgr'][...,::-1],interpolation='nearest');ins.set_xticks([]);ins.set_yticks([])
  for s in ins.spines.values():s.set_color('white');s.set_linewidth(1)
  ins.text(.0,1.025,'Pose-conditioned\nOrthophoto Observation',transform=ins.transAxes,fontsize=5.9,va='bottom',color=INK,bbox=dict(fc='white',ec='none',pad=1,alpha=.95))
 # Short interpretation under the panels: no unsupported causal claim.
 fig.text(.018,.178,'Landmark + candidate locations',fontsize=6.9)
 fig.text(.350,.178,'Language + geometry → global belief',fontsize=6.9)
 fig.text(.682,.178,r'$B_t=\mathrm{softmax}(s_0+\Delta_t)$',fontsize=8.1)
 fig.text(.018,.126,'Same world-XY crop in all panels; GT is analysis-only.',fontsize=6.0)
 fig.text(.682,.126,r'$\|B_7-B_0\|_1=0.0262$; final error 11.95 m',fontsize=6.5)
 cbax=fig.add_axes([.380,.080,.235,.022]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[1e-5,1e-3,1e-2]);cb.ax.tick_params(labelsize=5.8,pad=1,length=2);cb.outline.set_linewidth(.4);cb.solids.set_rasterized(False)
 fig.text(.497,.119,'Goal probability · shared log scale',ha='center',fontsize=6.0)
 fig.text(.018,.065,'● Candidates',color=ORANGE,fontsize=6.3)
 fig.text(.145,.065,'★ GT',color=RED,fontsize=6.3)
 fig.text(.218,.065,'— Path',color=BLUE,fontsize=6.3)
 fig.text(.682,.065,'ROI support ≠ target visibility',fontsize=6.0,color='#007782')
 fig.text(.018,.013,'CityNav / OpenStreetMap contributors • birmingham_block_5:8:4 • formal frozen C1',fontsize=5.6,color='#52626d')
 fig.text(.982,.013,'LOCAL RGB • redistribution unconfirmed' if args.include_rgb else 'PUBLIC • map geometry; no RGB pixels',ha='right',fontsize=5.4,color='#52626d')
 for ext in ['pdf','png','svg']:fig.savefig(out/f'geombelief_overview.{ext}',dpi=450,facecolor='white',metadata={'Creator':'GeomBelief real-evidence plotting script'} if ext=='pdf' else None)
 plt.close(fig)
 manifest={'episode_id':E['episode_id'],'include_rgb':args.include_rgb,'figure_inches':[7.2,3.0],'aspect_ratio':7.2/3.0,'display_world_bounds':v,'shared_color_norm':{'type':'log','vmin':norm.vmin,'vmax':norm.vmax,'alpha':.72,'under':'saturated to lower color; values unchanged'},'checkpoint_sha256':E['checkpoint_sha256'],'source_fields_sha256':E['source_array_sha256'],'source_evidence_sha256':sha(root/'OVERVIEW_EVIDENCE.json'),'script_sha256':sha(__file__),'outputs':{ext:sha(out/f'geombelief_overview.{ext}') for ext in ['pdf','png','svg']}}
 (out/'render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out)
if __name__=='__main__':main()
