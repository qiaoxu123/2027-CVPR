"""Posthoc TRAIN example selection. Public input forward then private evaluation overlay."""
import argparse,json,html,sys
from pathlib import Path
import cv2,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.unified_pgvb_v3 import UnifiedPGVBV3
from scripts.unified_pgvb_v3_pilot import V3CacheDataset,_collate,_load_base,_depth_patch_tensor
from scripts.diagnose_v3_observation_fusion import score,top_position
from graphnav.checkpointing import atomic_json,file_sha256

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for k in ('run','experiment','source-v2','data-root','sbf-source','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.sbf_source));from sbfnav.mapdata import GROUND_LEVEL
    rows=[json.loads(p.read_text()) for p in sorted((a.experiment/'full/rows').glob('*.json'))];assert len(rows)==4405
    def hit(r,name):return r['variants'][name]['fused']['hit']
    conditions=[('native_fusion_damage',lambda r:r['variants']['start_native']['geometry']['hit'] and not hit(r,'start_native')),
                ('planned_help',lambda r:not hit(r,'start_native') and hit(r,'plan5_yaw0')),
                ('planned_damage',lambda r:hit(r,'start_native') and not hit(r,'plan5_yaw0')),
                ('yaw0_help',lambda r:not hit(r,'start_native') and hit(r,'start_yaw0')),
                ('yaw0_damage',lambda r:hit(r,'start_native') and not hit(r,'start_yaw0')),
                ('landmark_gate',lambda r:r['anchor_count']==0 and r['variants']['start_native']['support']['raw_depth_region_support'])]
    chosen=[];seen=set()
    for tag,condition in conditions:
        for row in rows:
            physical=':'.join(row['episode_id'].split(':')[:2])
            if physical not in seen and condition(row):chosen.append((tag,row));seen.add(physical);break
    geom,_=_load_base(a.source_v2,'cuda');model=UnifiedPGVBV3(geom).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.run/'train/last.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    encoder=ObservedSigLIPMatchingEncoder(device='cuda',microbatch=1);renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'unused',image_size=384)
    ds=V3CacheDataset(a.run/'cache');a.output.mkdir(parents=True,exist_ok=True);examples=[]
    for tag,row in chosen:
        public,private=ds[row['index']];base,labels=_collate([(public,private)],'cuda');bounds=base['world_bounds'][0];ground=GROUND_LEVEL[row['map_name']]
        zero=np.asarray(row['start_pose']).copy();zero[3]=0
        cases=[]
        for name,pose in [('start_native',row['start_pose']),('start_yaw0',zero),('plan5_yaw0',public['frame_poses'][0].numpy())]:
            if name=='plan5_yaw0' and not row['anchor_count']:
                continue  # Padded cache frames are not real observations.
            obs=renderer.current_rgbd(row['map_name'],pose,ground)
            if name=='plan5_yaw0':kw=base
            else:
                image=encoder.processor(images=[cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB)],return_tensors='pt').to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):tokens=encoder.model.vision_model(**image).last_hidden_state[0].half().float()
                depth,valid=_depth_patch_tensor(obs)
                if not row['anchor_count']:valid.zero_()
                kw=dict(base,rgb_patches=tokens[None,None],depth_patches=depth.cuda().half().float()[None,None],patch_valid=valid.cuda()[None,None],
                        frame_poses=torch.as_tensor(np.asarray(pose),device='cuda',dtype=torch.float32)[None,None],footprints_world=torch.as_tensor(obs['footprint_world'],device='cuda')[None,None])
            out=model(**kw)
            target=labels['target_xy'][0]*(bounds[2:]-bounds[:2])+bounds[:2]
            metric=score(out.fused_logits[0],target,bounds)
            assert metric['hit']==row['variants'][name]['fused']['hit'] and abs(metric['NE']-row['variants'][name]['fused']['NE'])<.01
            filename=f"{tag}_{name}.png";assert cv2.imwrite(str(a.output/(tag+'_'+name+'_rgb.png')),obs['bgr'])
            fig,axes=plt.subplots(1,4,figsize=(17,4));axes[0].imshow(cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB));axes[0].set_title(name+' | native384');axes[0].axis('off')
            extent=[float(bounds[0]),float(bounds[2]),float(bounds[1]),float(bounds[3])]
            for ax,z,title in zip(axes[1:],(out.geometry_global,out.visual_global,out.fused_logits),('Geometry','Visual logits','Fused')):
                ax.imshow(z[0].cpu().numpy(),origin='lower',extent=extent,cmap='magma');ax.plot(*target.cpu().numpy(),'c*',ms=10)
                if private['target_polygon'] is not None:
                    poly=np.asarray(private['target_polygon']);ax.plot(*np.vstack((poly,poly[0])).T,'c--',lw=.7)
                for fp in kw['footprints_world'][0].cpu().numpy():ax.plot(*np.vstack((fp,fp[0])).T,'w-',lw=.5)
                if title!='Visual logits':ax.plot(*top_position(z[0],bounds).cpu().numpy(),'bx',ms=8)
                ax.set(xlim=extent[:2],ylim=extent[2:],title=title);ax.set_aspect('equal')
            fig.suptitle(f"{tag} | {row['episode_id']} | NE={metric['NE']:.2f}m | cyan: private posthoc point/weak Contour; white: footprint; blue: predicted")
            fig.tight_layout();fig.savefig(a.output/filename,dpi=120);plt.close(fig);cases.append(filename)
        examples.append(dict(kind=tag,episode_id=row['episode_id'],instruction=row['instruction'],images=cases))
    atomic_json(dict(checkpoint_sha256=file_sha256(a.run/'train/last.pt'),scope='TRAIN posthoc examples; planned5 RGB shows first view, belief uses five; weak Contour is not pixel GT',examples=examples),a.output/'examples.json')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>TRAIN观测配对诊断</h1><p>不是导航轨迹；规划五视点只显示首帧RGB；Contour为弱几何标注。</p>'+''.join('<h2>'+html.escape(r['kind']+' '+r['episode_id'])+'</h2><p>'+html.escape(r['instruction'])+'</p>'+''.join(f'<a href="{x}"><img width="1200" src="{x}"></a>' for x in r['images']) for r in examples))
    print(json.dumps(dict(examples=len(examples))))

if __name__=='__main__':main()
