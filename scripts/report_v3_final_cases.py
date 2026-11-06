"""Render actual final-checkpoint TRAIN cases selected only for posthoc diagnosis."""
import argparse, html, json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.unified_pgvb_v3 import UnifiedPGVBV3
from scripts.unified_pgvb_v3_pilot import V3CacheDataset, _collate, _load_base


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for k in ('run','source-v2','rgbd-dir','ground-levels','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    diagnosis=json.loads((a.run/'train_heldout_capacity.json').read_text())
    checkpoint=a.run/'train/last.pt'
    assert file_sha256(checkpoint)==diagnosis['summary']['checkpoint_sha256']
    ds=V3CacheDataset(a.run/'cache'); geom,_=_load_base(a.source_v2,'cuda')
    model=UnifiedPGVBV3(geom).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
    renderer=CityNavObservedViewRenderer(a.rgbd_dir,a.output/'unused',image_size=384)
    levels=json.loads(a.ground_levels.read_text()); levels=levels.get('ground_levels',levels)
    cases=[]
    for kind in ('fix','break','supported_failure'):
        def match(r):
            if kind=='fix':return not r['geometry_correct'] and r['fused_correct']
            if kind=='break':return r['geometry_correct'] and not r['fused_correct']
            return not r['geometry_correct'] and not r['fused_correct'] and r['correct_region_observed']
        for row in [r for r in diagnosis['rows'] if r['region_defined'] and match(r)][:2]:
            public,private=ds[row['index']]; kw,_=_collate([(public,private)],'cuda')
            out=model(**kw)  # GT remains outside the forward inputs.
            bounds=public['world_bounds'].numpy(); span=bounds[2:]-bounds[:2]
            gt=private['target_xy'].numpy()*span+bounds[:2]
            obs=renderer.current_rgbd(public['map_name'],public['frame_poses'][0].numpy(),levels[public['map_name']])
            name=f"{kind}_{row['index']:05d}"
            assert cv2.imwrite(str(a.output/(name+'_rgb.png')),obs['bgr'])
            fig,axes=plt.subplots(1,4,figsize=(17,4))
            axes[0].imshow(cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB))
            axes[0].set_title('TRAIN planned view 1/5; native384');axes[0].axis('off')
            for ax,z,title in zip(axes[1:],(out.geometry_global,out.visual_global,out.fused_logits),('Geometry','Visual logits','Fused')):
                values=z[0].cpu().numpy(); extent=[bounds[0],bounds[2],bounds[1],bounds[3]]
                ax.imshow(values,origin='lower',extent=extent,cmap='magma');ax.plot(*gt,'c*',ms=10)
                for fp in public['footprints_world'].numpy():ax.plot(*np.vstack((fp,fp[0])).T,'w-',lw=.4)
                if title!='Visual logits':
                    idx=int(z[0].flatten().topk(20).indices[0]); pred=bounds[:2]+np.array([(idx%384+.5)/384,(idx//384+.5)/384])*span
                    ax.plot(*pred,'bx',ms=9)
                ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),title=title);ax.set_aspect('equal')
            fig.suptitle(kind+' | '+private['episode_id']+' | cyan: posthoc GT; blue: predicted; white: observed footprints')
            fig.tight_layout();fig.savefig(a.output/(name+'.png'),dpi=130);plt.close(fig)
            cases.append(dict(**row,kind=kind,image=name+'.png',rgb=name+'_rgb.png'))
    atomic_json(dict(checkpoint_sha256=file_sha256(checkpoint),scope='TRAIN heldout posthoc selected examples; planned views, not navigation; no pixel instance GT',cases=cases),a.output/'cases.json')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>最终权重TRAIN真实RGB诊断</h1><p>公开地标规划的离线观测，不是闭环。GT只用于事后评分；不是实例识别准确率。</p>'+''.join(f'<h2>{html.escape(c["kind"]+": "+c["episode_id"])}</h2><a href="{c["image"]}"><img width="1200" src="{c["image"]}"></a><a href="{c["rgb"]}">原生RGB</a>' for c in cases))
    print(json.dumps(dict(cases=len(cases),checkpoint_sha256=file_sha256(checkpoint))))


if __name__=='__main__':main()
