"""Frozen TRAIN-heldout diagnostics; private GT is read only after forward."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.unified_pgvb_v3_pilot import V3CacheDataset,fit_dev_partition,_load_base,_collate
from graphnav.unified_pgvb_v3 import UnifiedPGVBV3,offline_topk
from graphnav.checkpointing import atomic_json,file_sha256

@torch.no_grad()
def inspect(model,ds,indices,control='normal'):
    model.eval();rows=[]
    supported=[i for i in indices if bool(ds[i][0]['patch_valid'].any())]
    donors={}
    public_metadata={i:ds[i][0] for i in supported}
    for j,i in enumerate(supported):
        src=public_metadata[i]
        candidates=supported[j+1:]+supported[:j]
        crossmap=[q for q in candidates if public_metadata[q]['map_name']!=src['map_name']]
        distant=[q for q in candidates if (public_metadata[q]['frame_poses'][0,:2]-src['frame_poses'][0,:2]).norm()>20]
        eligible=crossmap or distant
        if eligible: donors[i]=eligible[0]
    images={}
    for i in indices:
        public,private=ds[i];kw,labels=_collate([(public,private)],'cuda')
        if control=='zero_rgb':kw['rgb_patches']=torch.zeros_like(kw['rgb_patches'])
        if control=='shuffled_rgb' and i in donors:
            # Preserve recipient footprint, validity, pose, language and depth.
            kw['rgb_patches']=ds[donors[i]][0]['rgb_patches'][None].cuda().float()
        torch.cuda.synchronize();t=time.perf_counter();out=model(**kw);torch.cuda.synchronize()
        ms=(time.perf_counter()-t)*1000
        geo=offline_topk(out,labels['target_xy'],kw['world_bounds'],branch='geometry')
        fused=offline_topk(out,labels['target_xy'],kw['world_bounds'],branch='fused')
        span=kw['world_bounds'][0,2:]-kw['world_bounds'][0,:2]
        h,w=out.fused_logits.shape[-2:];yy,xx=torch.meshgrid((torch.arange(h,device='cuda')+.5)/h,(torch.arange(w,device='cuda')+.5)/w,indexing='ij')
        xy=torch.stack((xx,yy),-1);near=((xy-labels['target_xy'][0])*span).norm(dim=-1)<=20
        delta=out.fused_logits-out.geometry_global
        assert torch.equal(delta[~out.observed_mask],torch.zeros_like(delta[~out.observed_mask]))
        ix=(labels['target_xy'][0]*w).long().clamp(0,w-1)
        raw=out.geometry_logits[0];rid=int(raw.argmax());rxy=torch.tensor([(rid%28+.5)/28,(rid//28+.5)/28],device='cuda');native_ne=float(((rxy-labels['target_xy'][0])*span).norm())
        row=dict(index=i,episode_id=private['episode_id'],map_name=public['map_name'],control=control,donor=donors.get(i) if control=='shuffled_rgb' else None,
                 observed=bool(out.observed_mask.any()),center_observed=bool(out.observed_mask[0,ix[1],ix[0]]),support_fraction=float(out.observed_mask.float().mean()),
                 native_geometry_NE=native_ne,native_geometry_hit=native_ne<=20,geometry_NE=geo['Top1_NE_m'][0],fused_NE=fused['Top1_NE_m'][0],
                 inference_ms=ms,geometry_logit_range=float(raw.max()-raw.min()),visual_delta_max=float(delta.abs().max()))
        for branch,metric in [('geometry',geo),('fused',fused)]:
            for k in [1,5,20]:row[f'{branch}_hit{k}']=metric['hit_flags'][str(k)][0]
            logits=out.geometry_global if branch=='geometry' else out.fused_logits
            row[f'{branch}_mass20']=float((logits.flatten().softmax(0).reshape(h,w)*near).sum())
        rows.append(row)
        # Keep limited genuine model fields for posthoc heatmaps, no influence on inference.
        if control=='normal':images[i]=(out.geometry_global[0].cpu().numpy(),out.visual_global[0].cpu().numpy(),out.fused_logits[0].cpu().numpy(),out.observed_mask[0].cpu().numpy())
    def summary(items):
        return dict(n=len(items),**{f'{b}_R{k}':float(np.mean([v[f'{b}_hit{k}'] for v in items])) for b in ['geometry','fused'] for k in [1,5,20]},
                    geometry_NE=float(np.mean([v['geometry_NE'] for v in items])),fused_NE=float(np.mean([v['fused_NE'] for v in items])),
                    fixes=sum(not v['geometry_hit1'] and v['fused_hit1'] for v in items),breaks=sum(v['geometry_hit1'] and not v['fused_hit1'] for v in items),
                    inference_ms_excluding_first=float(np.mean([v['inference_ms'] for v in items[1:]])) if len(items)>1 else None)
    metrics=dict(all=summary(rows),by_map={k:summary([x for x in rows if x['map_name']==k]) for k in sorted(set(x['map_name'] for x in rows))})
    for label,predicate in [('observed',lambda x:x['observed']),('center_observed',lambda x:x['center_observed']),('no_visual',lambda x:not x['observed'])]:
        rr=[x for x in rows if predicate(x)];metrics[label]=summary(rr) if rr else None
    return dict(metrics=metrics,rows=rows),images

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--source-v2',type=Path,required=True);p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rgbd-dir',type=Path);p.add_argument('--ground-levels',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);ds=V3CacheDataset(a.cache);fit,dev=fit_dev_partition(ds,42)
    torch.manual_seed(42);geom,_=_load_base(a.source_v2,'cuda');model=UnifiedPGVBV3(geom).cuda()
    before,_=inspect(model,ds,dev);atomic_json(before,a.output/'before.json')
    checkpoint=torch.load(a.training/'last.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['model'],strict=True)
    results={};arrays={}
    for control in ['normal','zero_rgb','shuffled_rgb']:
        results[control],imgs=inspect(model,ds,dev,control);atomic_json(results[control],a.output/f'{control}.json')
        if control=='normal':arrays=imgs
    normals=results['normal']['rows'];chosen=[]
    for kind in ['fix','break','both_wrong','both_right']:
        def match(v):
            g,f=v['geometry_hit1'],v['fused_hit1']
            return {'fix':not g and f,'break':g and not f,'both_wrong':not g and not f,'both_right':g and f}[kind]
        chosen.extend([v['index'] for v in normals if match(v)][:2])
    renderer=None
    if a.rgbd_dir and a.ground_levels:
        from graphnav.renderer import CityNavObservedViewRenderer
        renderer=CityNavObservedViewRenderer(a.rgbd_dir,a.output/'unused',image_size=384)
        levels=json.loads(a.ground_levels.read_text());levels=levels.get('ground_levels',levels)
    figlist=[]
    for index in chosen:
        public,private=ds[index];geo,vis,fused,support=arrays[index];bounds=public['world_bounds'].numpy();gt=private['target_xy'].numpy()*(bounds[2:]-bounds[:2])+bounds[:2]
        fig,axes=plt.subplots(1,4 if renderer else 3,figsize=(17 if renderer else 13,4));axs=axes[-3:];extent=[bounds[0],bounds[2],bounds[1],bounds[3]]
        if renderer:
            import cv2
            pose=public['frame_poses'][0].numpy()
            observation=renderer.current_rgbd(public['map_name'],pose,levels[public['map_name']])
            bgr=observation['bgr'];cv2.imwrite(str(a.output/f'rgb_{index:04d}.png'),bgr)
            axes[0].imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));axes[0].set_title('Actual native384 RGB; weak contour')
            fp=public['footprints_world'][0].numpy();basis=np.stack([fp[1]-fp[0],fp[3]-fp[0]],axis=1)
            polygon=private['target_polygon']
            if polygon is not None:
                uv=np.linalg.solve(basis,(np.asarray(polygon)-fp[0]).T).T*383
                axes[0].plot(*np.vstack([uv,uv[0]]).T,'c-',lw=.7)
            axes[0].set_xlim(0,383);axes[0].set_ylim(383,0)
        for ax,z,title in zip(axs,[geo,vis,fused],['Geometry','Visual evidence','Fused']):
            ax.imshow(z,origin='lower',extent=extent,cmap='magma');ax.plot(*gt,'c*',ms=9)
            if support.any() and not support.all():ax.contour(support,levels=[.5],extent=extent,origin='lower',colors='lime',linewidths=.4)
            for fp in public['footprints_world'].numpy():ax.plot(*np.vstack([fp,fp[0]]).T,'w-',lw=.4)
            if title!='Visual evidence':
                y,x=np.unravel_index(torch.from_numpy(z).flatten().topk(20).indices[0].item(),z.shape);pred=bounds[:2]+np.array([(x+.5)/384,(y+.5)/384])*(bounds[2:]-bounds[:2]);ax.plot(*pred,'bx',ms=7)
            ax.set_title(title);ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal')
        fig.suptitle(private['episode_id']+' | cyan GT, blue prediction, green support');fig.tight_layout();fn=f'case_{index:04d}.png';fig.savefig(a.output/fn,dpi=140);plt.close(fig);figlist.append(fn)
    atomic_json(dict(source_v2_sha256=file_sha256(a.source_v2),checkpoint_sha256=file_sha256(a.training/'last.pt'),fit=len(fit),dev=len(dev),before=before['metrics'],after={k:v['metrics'] for k,v in results.items()},figures=figlist,scope='TRAIN physical-target heldout in this new training; initial V2 trained on TRAIN; no navigation'),a.output/'summary.json')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V3 TRAIN diagnostic</title><h1>V3 TRAIN诊断；不是导航SR</h1>'+''.join(f'<a href="{x}"><img width="1100" src="{x}"></a>' for x in figlist))
    print(json.dumps({k:v['metrics']['all'] for k,v in results.items()},indent=2))
if __name__=='__main__':main()
