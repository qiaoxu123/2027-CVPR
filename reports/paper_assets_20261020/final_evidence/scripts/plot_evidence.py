"""New P0 temporal/alignment panels; never redraw the four existing figures.

RGB-containing exports have a hard-coded local_only destination. Public output
is a numerical time-series diagnostic, with explicit model/control identities.
"""
import textwrap
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
from common import *

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'real':'#1676b6','zero':'#7b7b7b','shuffle':'#ba54a1','no-belief':'#d3821b'}


def export(fig,path,metadata):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)
    dump(path.with_suffix('.json'),metadata|{'sha256':{e:sha(path.with_suffix('.'+e)) for e in ['pdf','png']}})


def local_alignment():
    index=read(FINAL/'RGB_FRAME_INDEX.json');allcases={r['case_id']:r for r in cases()}
    frames={(r['case_id'],r['tensor_index']):r for r in index['frames']}
    for cid in SELECTED:
        c=allcases[cid];a=fields(cid);world=a['world_xy'];gt=np.asarray(c['target_xyz'][:2]);selection=index['selection'][cid]
        ids=selection['tensor_indices'];poses=np.asarray(c['trace']['poses']);p=[probability(a['real_belief'][i]) for i in ids]
        norm=LogNorm(vmin=max(min(float(v.max()) for v in p)*1e-4,1e-8),vmax=max(float(v.max()) for v in p))
        limit=max(float(np.abs(a['real_residual'][ids]).max()),1e-8)
        fig,axes=plt.subplots(3,3,figsize=(10.0,9.0),layout='constrained')
        for col,(name,i) in enumerate(zip(['Start','Key observation','Terminal decision'],ids)):
            row=frames[(cid,i)];assert row['exact_archive_crop_match']
            with np.load(FINAL/row['local_npz']) as obs:bgr=obs['bgr']
            ax=axes[0,col];ax.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));ax.set_axis_off()
            pix=row['target_xy_projected_pixel']
            if row['target_xy_inside_footprint']:ax.scatter(*pix,marker='+',c='#33ff77',s=35,lw=1)
            proxy='projected inside' if row['target_xy_inside_footprint'] else 'projected outside'
            ax.set_title(f'{name}: step {row["step"]}\nExact model-input RGB; target XY {proxy}',fontsize=8)
            ax=axes[1,col];im=ax.scatter(world[:,0],world[:,1],c=p[col],s=8,cmap='magma',norm=norm,marker='s',rasterized=False)
            ax.plot(poses[:,0],poses[:,1],c='.75',lw=.7,label='Full executed path (annotation)')
            ax.plot(poses[:row['step']+1,0],poses[:row['step']+1,1],c='#22a4f0',lw=1.2)
            ax.scatter(*poses[row['step'],:2],s=18,c='#22a4f0',marker='^')
            ax.scatter(*gt,marker='*',s=55,c='#35dc6e',edgecolors='black',lw=.4)
            top=world[p[col].argmax()];ax.scatter(*top,marker='X',s=35,c='#ffbd36',edgecolors='black',lw=.4)
            ax.add_patch(Polygon(row['footprint_world'],fill=False,edgecolor='#14a5b2',ls='--',lw=.7))
            error=np.linalg.norm(top-gt);ax.set_title(f'Goal belief at the same step | error {error:.2f} m',fontsize=8)
            ax.set(xlabel='World X (m)',ylabel='World Y (m)',aspect='equal')
            ax=axes[2,col];delta=a['real_residual'][i]
            dm=ax.scatter(world[:,0],world[:,1],c=delta,s=8,cmap='RdBu_r',vmin=-limit,vmax=limit,marker='s')
            ax.scatter(*gt,marker='*',s=55,c='#35dc6e',edgecolors='black',lw=.4)
            ax.set(xlabel='World X (m)',ylabel='World Y (m)',aspect='equal')
            ax.set_title(f'Applied residual\nmax |delta| = {abs(delta).max():.4f} logits',fontsize=8)
        fig.colorbar(im,ax=axes[1,:],shrink=.7,pad=.02,label='Goal probability (shared log scale)')
        fig.colorbar(dm,ax=axes[2,:],shrink=.7,pad=.02,label='Applied logit residual (shared symmetric scale)')
        identity='Formal main checkpoint' if c['model_identity']=='formal' else 'Separate calibrated residual on frozen formal base'
        fig.suptitle(f'{identity} | {cid}\n'+textwrap.fill(c['instruction'],115),fontsize=9)
        fig.supxlabel('Green star: GT; yellow X: dense Top-1; blue: executed prefix/UAV; gray: full executed path; cyan: observed footprint.\nACTUAL ORTHOPHOTO MODEL INPUT, NOT ONBOARD PERSPECTIVE FPV. Target + is XY projection only, not visibility.\nCityNav / SensatUrban. Local research review only; no RGB redistribution clearance.',fontsize=7)
        export(fig,FINAL/'local_only/figures'/f'p0_{cid}_rgb_alignment',{
            'case_id':cid,'episode_id':c['episode_id'],'model_identity':c['model_identity'],
            'base_checkpoint_sha256':c['checkpoint_sha256'],'residual_checkpoint_sha256':c['residual_checkpoint_sha256'],
            'frame_steps':selection['steps'],'frame_npz_sha256':[frames[(cid,i)]['npz_sha256'] for i in ids],
            'fields_sha256':sha(ASSETS/'fields'/f'{cid}.npz'),'rgb_public':False,
            'alignment':'exact archived frame step, pose, crop hash and tensor index; no neighboring-frame substitution'})
        print('local RGB figure',cid,flush=True)


def public_diagnostics():
    cs={r['case_id']:r for r in cases()};fig,axes=plt.subplots(2,3,figsize=(10.5,5.4),layout='constrained')
    metadata=[]
    for col,cid in enumerate(SELECTED):
        c=cs[cid];a=fields(cid);xy=a['world_xy'];distance=np.linalg.norm(xy-np.asarray(c['target_xyz'][:2]),axis=1)
        meta=read(ASSETS/'fields'/f'{cid}.json');steps=[r['step'] for r in meta['frames']]
        controls=['real'] if c['model_identity']=='formal' else ['real','zero','shuffle','no-belief']
        ax=axes[0,col]
        for control in controls:
            errors=distance[a[control+'_belief'].argmax(1)]
            ax.plot(steps,errors,color=COLORS[control],lw=1.3,label=control,ls='--' if control in ['zero','no-belief'] else '-')
        ax.axhline(20,c='.2',ls=':',lw=.8,label='20 m goal radius');ax.legend(fontsize=6,loc='best')
        ax.set(xlabel='Recorded observation step',ylabel='Dense Top-1 goal error (m)',title=['Formal main success','CVR correction','CVR degradation'][col])
        ax=axes[1,col];l1=[];dh=[]
        for s,b in zip(a['real_static'],a['real_belief']):
            ps,pb=probability(s),probability(b);l1.append(abs(ps-pb).sum());dh.append(entropy(pb)-entropy(ps))
        ax.plot(steps,l1,c='#1676b6',marker='.',lw=1.2,label='L1(static, Real)')
        second=ax.twinx();second.plot(steps,dh,c='#d3821b',ls='--',lw=1,label='Entropy difference')
        ax.set(xlabel='Recorded observation step',ylabel='Belief L1 (unitless)');second.set_ylabel('Entropy change (nats)',color='#a5681d')
        ax.set_title(':'.join(map(str,c['episode_id'])),fontsize=8)
        metadata.append({'case_id':cid,'episode_id':c['episode_id'],'model_identity':c['model_identity'],
            'base_checkpoint_sha256':c['checkpoint_sha256'],'residual_checkpoint_sha256':c['residual_checkpoint_sha256'],
            'source_fields_sha256':sha(ASSETS/'fields'/f'{cid}.npz'),'steps':steps})
    fig.suptitle('Measured temporal visual-belief effects at archived Real-pose prefixes',fontsize=10)
    fig.supxlabel('Top-1 = dense-logit argmax; controls are matched-path belief interventions, not their independent rollout paths.\nLeft: formal model. Middle/right: separate CVR checkpoint. No RGB, raster, or scene pixels in this figure.',fontsize=7)
    export(fig,FINAL/'figures/p0_temporal_visual_evidence',{'cases':metadata,'rgb_pixels':False,
        'source_statistics':'VISUAL_CAUSAL_CASES.csv','units':{'error':'meters','L1':'unitless','entropy_change':'nats'},
        'limits':'Input/control sensitivity is not proof of semantic understanding. Only annotated selected cases; no population frequency claim.'})


if __name__=='__main__':
    local_alignment();public_diagnostics()
