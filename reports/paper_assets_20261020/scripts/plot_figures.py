"""Publication figures from real cached fields/traces; RGB remains local by default."""
from common import *
import argparse,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,TwoSlopeNorm
from matplotlib.patches import Polygon,Circle
from shapely.geometry import Polygon as ShapePolygon,Point,box

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def probability(v):
    v=np.asarray(v,dtype=np.float64);p=np.exp(v-v.max());return p/p.sum()

def field_grid(a):
    xy=a['world_xy'];n=int(np.sqrt(len(xy)));return xy.reshape(n,n,2)

def polygons(record):
    bounds=np.asarray(record['world_bounds']);lo=bounds[:2];span=bounds[2:]-lo
    return [(l,lo+np.asarray(l['contour'])*span) for l in record['landmarks'] if len(l.get('contour',[]))>=3]

def base_map(ax,record,labels=False):
    context=CONTEXT.get(record['map_name'],[])
    for item in context:
        xy=np.asarray(item['contour'])
        if len(xy)>=3:ax.add_patch(Polygon(xy,closed=True,facecolor='#d6dde3',edgecolor='#9ba9b2',lw=.3,alpha=.16))
    for i,(l,xy) in enumerate(polygons(record)):
        road='road' in l.get('object_type','').lower()
        ax.add_patch(Polygon(xy,closed=True,facecolor='#c8d6df' if road else '#d5e8d2',edgecolor='#617b8b',lw=.55,alpha=.55))
    if labels:
        used=set()
        for l,xy in polygons(record):
            if l['name'] in used:continue
            used.add(l['name']);bounds=record['world_bounds'];visible=ShapePolygon(xy).buffer(0).intersection(box(*bounds))
            if visible.is_empty:continue
            center=np.asarray(visible.centroid.coords[0]);right=center[0]>bounds[0]+.7*(bounds[2]-bounds[0])
            ax.annotate(l['name'],center,fontsize=6,ha='right' if right else 'left',bbox=dict(fc='white',alpha=.8,ec='none',pad=1),clip_on=True)
    b=record['world_bounds'];ax.set(xlim=(b[0],b[2]),ylim=(b[1],b[3]),aspect='equal',xlabel='World X (m)',ylabel='World Y (m)')
    ax.tick_params(labelsize=6);ax.grid(alpha=.15,lw=.4)

def markers(ax,case,top=None,path=True,footprints=None,legend=False):
    gt=np.asarray(case['target_xyz'])[:2];poses=np.asarray(case['trace']['poses'])
    ax.scatter(*gt,c='#32d474',marker='*',edgecolor='black',lw=.4,s=85,zorder=10,label='GT goal')
    ax.add_patch(Circle(gt,20,fill=False,color='#32b36d',lw=.9,ls='--'))
    if path:
        ax.plot(poses[:,0],poses[:,1],'-',c='#248fe0',lw=1.4,label='Executed UAV path',zorder=7)
        ax.scatter(*poses[0,:2],c='#2385ca',marker='^',s=35,zorder=9,label='Start')
        ax.scatter(*poses[-1,:2],c='#dc5a4b',marker='s',s=28,zorder=9,label='Final position')
    if top is not None:ax.scatter(*top,c='#ffbe3d',edgecolor='black',lw=.6,marker='X',s=48,zorder=11,label='Predicted goal')
    if footprints:
        for frame in footprints:
            xy=np.asarray(frame['footprint_world']);ax.add_patch(Polygon(xy,fill=False,edgecolor='#22b8b3',lw=.65,alpha=.5))
    if legend:ax.legend(fontsize=6,loc='best',framealpha=.85)

def heat(ax,case,a,p,norm,title,path=False,footprints=None):
    grid=field_grid(a);n=grid.shape[0]
    im=ax.pcolormesh(grid[...,0],grid[...,1],p.reshape(n,n),shading='nearest',cmap='magma',norm=norm,rasterized=False)
    base_map(ax,case['public_map'])
    top=a['world_xy'][p.argmax()];markers(ax,case,top,path=path,footprints=footprints)
    error=np.linalg.norm(top-np.asarray(case['target_xyz'])[:2]);ax.set_title(title+f'\nTop-1 error {error:.2f} m')
    return im

def rgb_or_footprint(ax,case,frame,rgb):
    if rgb:
        p=OUT/'local_only/observations'/f"{case['case_id']}_step{frame['step']}.npz"
        obs=np.load(p);ax.imshow(obs['bgr'][...,::-1]);ax.axis('off');ax.set_title(f"Actual observed crop, t={frame['step']}")
    else:
        base_map(ax,case['public_map'])
        xy=np.asarray(frame['footprint_world']);ax.add_patch(Polygon(xy,fc='#30c5ba',ec='#078779',alpha=.25,lw=1))
        ax.scatter(*frame['pose'][:2],c='#087caf',marker='^',s=35)
        ax.set_title(f"Actual observation footprint, t={frame['step']}")

def save(fig,name,folder,case_ids,extra=None):
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):fig.savefig(folder/f'{name}.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    selected=[CASES[k] for k in case_ids if k in CASES]
    meta=dict(figure=name,cases=[{k:c[k] for k in ('case_id','episode_id','split','dataset_version','dataset_sha256','checkpoint_sha256','residual_checkpoint_sha256','source_predictions','source_predictions_sha256')} for c in selected],
              command='python reports/paper_assets_20261020/scripts/plot_figures.py'+(' --include-rgb' if RGB else ''),
              data_kind='Measured CityNav map, frozen prediction fields, executed trajectories; no generated scene',
              source_bundle='cases.jsonl; replay_manifest.json; fields/*.npz and fields/*.json; geometry_case.json',
              public_map_context_sha256=sha(OUT/'public_map_context.json'),observation_index='observation_index.json',
              image_use='Local-only real orthophoto crop previews; do not upload pending underlying image terms' if RGB else 'No RGB or raw raster pixels. Derived map/prediction figure. CityNav CC BY 4.0; map attribution: OpenStreetMap contributors (https://www.openstreetmap.org/copyright).',
              sha256={ext:sha(folder/f'{name}.{ext}') for ext in ('pdf','png')},**(extra or {}))
    dump(folder/f'{name}.json',meta)

def overview(folder):
    case=CASES['formal_val_unseen_success'];a=np.load(OUT/'fields'/f"{case['case_id']}.npz");meta=read(OUT/'fields'/f"{case['case_id']}.json")
    ps=probability(a['real_static'][0]);pb=probability(a['real_belief'][-1]);res=a['real_residual'][-1]
    norm=LogNorm(vmin=max(min(ps.min(),pb.min()),max(ps.max(),pb.max())*1e-5),vmax=max(ps.max(),pb.max()))
    fig=plt.figure(figsize=(17.2,6.5),layout='constrained');outer=fig.add_gridspec(1,4,width_ratios=[1.05,1,1.65,1.05])
    titles=['1  Language + Geographic Map','2  Geometric Goal Belief','3  Observation-Conditioned Visual Refinement','4  Goal-Directed Navigation']
    left=outer[0].subgridspec(2,1,height_ratios=[.38,1]);ax=fig.add_subplot(left[0]);ax.axis('off')
    ax.text(0,1,'Instruction (verbatim):\n'+textwrap.fill(case['instruction'],46),va='top',fontsize=9)
    ax=fig.add_subplot(left[1]);base_map(ax,case['public_map'],labels=True);markers(ax,case,path=False);ax.set_title('Actual named landmark / road polygons')
    middle=outer[1].subgridspec(2,1,height_ratios=[1,.4]);ax=fig.add_subplot(middle[0]);im=heat(ax,case,a,ps,norm,'Initial static belief B0')
    fig.colorbar(im,ax=ax,shrink=.6,label='Goal probability (shared log scale)')
    candidate=a['world_xy'][ps.argmax()]
    ref,poly=min(polygons(case['public_map']),key=lambda item:ShapePolygon(item[1]).distance(Point(candidate)))
    shape=ShapePolygon(poly);center=np.asarray(shape.centroid.coords[0]);displacement=candidate-center
    centroid_distance=np.linalg.norm(displacement);boundary_distance=shape.boundary.distance(Point(candidate))*(-1 if shape.covers(Point(candidate)) else 1)
    angle=np.degrees(np.arctan2(displacement[1],displacement[0]))
    ax=fig.add_subplot(middle[1]);ax.axis('off');ax.text(.02,.95,f'Measured grid: 28 x 28\nCandidate relative to {ref["name"]}:\nCentroid distance: {centroid_distance:.2f} m\nAngle from +X: {angle:.1f} deg\nSigned boundary distance: {boundary_distance:.2f} m\n\nWorld X/Y; north is not assumed.',va='top',fontsize=8)
    right=outer[2].subgridspec(2,2)
    for j,frame in enumerate([meta['frames'][0],meta['frames'][-1]]):rgb_or_footprint(fig.add_subplot(right[0,j]),case,frame,RGB)
    ax=fig.add_subplot(right[1,0]);heat(ax,case,a,pb,norm,'Final observed belief Bt',path=True)
    ax=fig.add_subplot(right[1,1]);grid=field_grid(a);im=ax.pcolormesh(grid[...,0],grid[...,1],res.reshape(28,28),cmap='RdBu_r',vmin=-.5,vmax=.5,shading='nearest');base_map(ax,case['public_map']);markers(ax,case,path=False);ax.set_title('Measured visual/history residual')
    fig.colorbar(im,ax=ax,shrink=.65,label='Logit correction')
    last=outer[3].subgridspec(2,1,height_ratios=[1,.4]);ax=fig.add_subplot(last[0]);base_map(ax,case['public_map']);top=case['trace']['step_diagnostics'][-1]['selected_goal_xy'];markers(ax,case,top,path=True,footprints=meta['frames'][::max(1,len(meta['frames'])//3)],legend=True);ax.set_title('Actual goal selection and Stop location')
    ax=fig.add_subplot(last[1]);ax.axis('off');m=case['trace']['metrics'];l1=np.abs(ps-pb).sum()
    ax.text(.02,.95,f"Episode {':'.join(map(str,case['episode_id']))}\nSR={int(m['SR'])}; NE={m['NE']:.2f} m\nTermination: {case['trace']['termination']}\nBelief L1(B0, Bt)={l1:.5f}\nFrozen C epoch1 (00f2acc0...)\nIllustrative success, not a visual-gain claim.",va='top',fontsize=8)
    for i,title in enumerate(titles):
        # Column headers are figure-level to keep nested axes unobstructed.
        bounds=outer[i].get_position(fig);fig.text((bounds.x0+bounds.x1)/2,1.015,title,ha='center',va='bottom',fontsize=10,fontweight='bold')
    fig.supxlabel('Real recorded episode. RGB shown only in local preview; public version uses measured observation footprints. GT is evaluation-only. Map: OpenStreetMap contributors / CityNav.',fontsize=7)
    save(fig,'figure1_overview',folder,[case['case_id']])

def geometry(folder):
    meta=read(OUT/'geometry_case.json');a=np.load(OUT/meta['arrays']);record=meta['public_map'];xy=a['world_xy'];grid=xy.reshape(28,28,2);gt=a['target_xy']
    ps={k:probability(a[k]) for k in ('name_type','shuffled','correct')};maximum=max(p.max() for p in ps.values());minimum=max(maximum*1e-5,min(p.min() for p in ps.values()));norm=LogNorm(minimum,maximum)
    fig=plt.figure(figsize=(14,7.4),layout='constrained');gs=fig.add_gridspec(2,3,height_ratios=[1,.43]);axes=[]
    relation_rows=[]
    for col,(arm,title) in enumerate([('name_type','Name / Type Only'),('shuffled','Shuffled Geometry'),('correct','Correct Geometry')]):
        ax=fig.add_subplot(gs[0,col]);axes.append(ax);p=ps[arm];im=ax.pcolormesh(grid[...,0],grid[...,1],p.reshape(28,28),shading='nearest',cmap='magma',norm=norm)
        base_map(ax,record,labels=arm=='correct');top=xy[p.argmax()]
        ax.scatter(*gt,c='#37e57c',edgecolor='black',lw=.4,s=110,marker='*',zorder=10,label='GT goal')
        ax.add_patch(Circle(gt,20,fill=False,ec='#37d477',ls='--',lw=.8))
        ax.scatter(*top,c='#ffbd2f',edgecolor='black',s=65,marker='X',zorder=11,label='Predicted goal')
        if arm=='shuffled':
            for _,poly in polygons(meta['donor_map']):ax.add_patch(Polygon(poly,fill=False,edgecolor='#26d7e5',ls=':',lw=.6))
        ax.set_title(f"{title}\nTop-1 error {np.linalg.norm(top-gt):.2f} m")
        valid=[(l,poly) for l,poly in polygons(record)]
        landmark,poly=min(valid,key=lambda v:ShapePolygon(v[1]).distance(Point(top)))
        shape=ShapePolygon(poly);center=np.asarray(shape.centroid.coords[0]);boundary=shape.boundary.distance(Point(top));distance=np.linalg.norm(top-center);angle=np.degrees(np.arctan2(*(top-center)[::-1]))
        signed=-boundary if shape.covers(Point(top)) else boundary
        ax.plot([center[0],top[0]],[center[1],top[1]],'--',c='#3de1ee',lw=.9)
        relation_rows.append([title,landmark['name'],f'{distance:.2f}',f'{angle:.1f}',f'{signed:.2f}'])
    axes[0].legend(fontsize=7,loc='lower left');fig.colorbar(im,ax=axes,shrink=.75,label='Goal probability (shared log scale)')
    ax=fig.add_subplot(gs[1,:]);ax.axis('off');table=ax.table(cellText=relation_rows,colLabels=['Candidate prediction','Nearest referenced landmark','Centroid distance (m)','Angle from +X (deg)','Signed boundary distance (m)'],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.6)
    fig.suptitle('Language-grounded map geometry: actual fixed-seed predictions\n'+textwrap.fill(record['instruction'],135),fontsize=11)
    fig.supxlabel('Fixed seed 0; separately trained matched-budget ablation heads. Outcome-selected illustration, not a same-weight intervention. Solid outlines: recipient map. Dotted cyan: shuffled donor map. Angles are mathematical, not verified compass bearings.\nMap: OpenStreetMap contributors / CityNav. Episode '+str(meta['episode_id']),fontsize=7)
    csvwrite(OUT/'candidate_landmark_relations.csv',[dict(zip(['arm','reference','centroid_distance_m','angle_from_plus_x_deg','signed_boundary_distance_m'],r)) for r in relation_rows])
    save(fig,'figure2_geometry',folder,[],dict(geometry_case='geometry_case.json',episode_id=meta['episode_id'],split=meta['split'],dataset_version=meta['dataset_version'],source_query_sha256=meta['source_query_sha256'],sources=meta['sources']))

def visual(folder):
    names=['visual_corrected','visual_reinforced','visual_degraded'];fig,axes=plt.subplots(3,7,figsize=(22,10.5),layout='constrained')
    summary=[]
    for row,name in enumerate(names):
        case=CASES[name];a=np.load(OUT/'fields'/f'{name}.npz');meta=read(OUT/'fields'/f'{name}.json');xy=a['world_xy'];gt=np.asarray(case['target_xyz'])[:2]
        ps=probability(a['real_static'][0]);pbs={k:probability(a[f'{k}_belief'][-1]) for k in ('real','zero','shuffle','no-belief')};p=pbs['real'];diff=p-ps
        maximum=max(ps.max(),*(v.max() for v in pbs.values()));minimum=max(maximum*1e-5,min(ps.min(),*(v.min() for v in pbs.values())));norm=LogNorm(minimum,maximum)
        rgb_or_footprint(axes[row,0],case,meta['frames'][-1],RGB)
        axes[row,0].set_ylabel(['A  Corrected','B  Reinforced','C  Degraded'][row]+'\nWorld Y (m)')
        im=heat(axes[row,1],case,a,ps,norm,'Initial belief B0')
        heat(axes[row,2],case,a,p,norm,'Real RGB: final Bt',path=True,footprints=[meta['frames'][-1]])
        ax=axes[row,3];grid=field_grid(a);limit=max(float(np.abs(diff).max()),1e-12);delta=ax.pcolormesh(grid[...,0],grid[...,1],diff.reshape(28,28),shading='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit)
        base_map(ax,case['public_map']);markers(ax,case,path=False);ax.set_title(f'Probability difference Bt - B0\nL1={np.abs(diff).sum():.4f}')
        fig.colorbar(delta,ax=ax,shrink=.6,label='Probability change')
        for col,(control,title) in enumerate([('zero','Zero RGB'),('shuffle','Shuffle RGB'),('no-belief','No residual')],start=4):heat(axes[row,col],case,a,pbs[control],norm,title)
        fig.colorbar(im,ax=[axes[row,i] for i in [1,2,4,5,6]],shrink=.65,label='Probability (shared row log scale)')
        mask=np.linalg.norm(xy-gt,axis=1)<=20
        summary.append(dict(case_id=name,episode_id=':'.join(map(str,case['episode_id'])),initial_error_m=float(np.linalg.norm(xy[ps.argmax()]-gt)),final_error_m=float(np.linalg.norm(xy[p.argmax()]-gt)),initial_GT20_probability=float(ps[mask].sum()),final_GT20_probability=float(p[mask].sum()),belief_L1=float(np.abs(diff).sum()),KL_B0_to_Bt_nats=float(np.sum(ps*np.log(ps/p))),top1_shift_m=float(np.linalg.norm(xy[ps.argmax()]-xy[p.argmax()])),navigation_SR=case['trace']['metrics']['SR'],navigation_NE_m=case['trace']['metrics']['NE']))
    fig.suptitle('Actual observation-conditioned belief updates: correction, reinforcement and failure\nSeparate calibrated residual checkpoint; all controls replay the same Real-RGB executed pose prefixes',fontsize=12)
    fig.supxlabel('GT star / dashed 20m circle; prediction X; blue executed path; cyan observation footprint. Per-row shared probability scale; difference scale is explicit.\nCases selected for illustration in source order; Real need not outperform Shuffle in an individual case. No-belief disables the added residual. Map: OpenStreetMap contributors / CityNav.',fontsize=8)
    csvwrite(OUT/'visual_case_summary.csv',summary);save(fig,'figure3_visual_updates',folder,names)

def navigation(folder):
    names=['formal_val_seen_success','formal_val_unseen_success','formal_val_seen_wrong_goal_false_stop','formal_val_unseen_wrong_goal_false_stop','formal_val_seen_visited_but_failed','formal_val_unseen_visited_but_failed']
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for ax,name in zip(axes.flat,names):
        case=CASES[name];m=case['trace']['metrics'];meta=read(OUT/'fields'/f'{name}.json');base_map(ax,case['public_map'],labels=True)
        markers(ax,case,case['trace']['step_diagnostics'][-1]['selected_goal_xy'],footprints=[meta['frames'][-1]],legend=True)
        label='Success' if m['SR'] else ('Visited but failed' if m['OSR'] else 'Wrong goal / false Stop')
        ax.set_title(f"{label} | {case['split']} | episode {case['index']}\nNE={m['NE']:.2f}m, SR={int(m['SR'])}, OSR={int(m['OSR'])}; {case['trace']['termination']}")
        ax.text(.01,-.23,textwrap.fill(case['instruction'],65),transform=ax.transAxes,fontsize=7,va='top')
    fig.suptitle('Archived formal model: two successes and four failures\nFrozen GraphNav-V1 C epoch1 (GeomBelief manuscript identity), checkpoint SHA 00f2acc0...',fontsize=12)
    fig.supxlabel('First qualifying episode per split/outcome in fixed source order; no artificial paths. Stars: GT; yellow X: final selected goal; red square: actual final position.\nVisited-but-failed = OSR=1 and SR=0; this alone does not establish a learned-Stop causal failure. Map: OpenStreetMap contributors / CityNav.',fontsize=8,y=-.03)
    save(fig,'figure4_navigation',folder,names)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-rgb',action='store_true');parser.add_argument('--output-dir',type=Path);args=parser.parse_args();RGB=args.include_rgb
    CASES={c['case_id']:c for c in jsonl(OUT/'cases.jsonl')}
    CONTEXT=read(OUT/'public_map_context.json')['maps']
    folder=args.output_dir or OUT/('local_only/figures' if RGB else 'figures')
    for function in (overview,geometry,visual,navigation):function(folder);print(function.__name__,'done',flush=True)
