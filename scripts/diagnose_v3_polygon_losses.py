"""Private TRAIN weak-label audit and autograd inspection, without optimizer updates."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from shapely.geometry import Polygon,Point
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.unified_pgvb_v3 import UnifiedPGVBV3,v3_losses
from scripts.unified_pgvb_v3_pilot import V3CacheDataset,fit_dev_partition,_collate,_load_base

def main():
    p=argparse.ArgumentParser()
    for key in ('run','source-v2','objects','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();ds=V3CacheDataset(a.run/'cache');_,dev=fit_dev_partition(ds,42)
    objects=json.loads(a.objects.read_text());rows=[];candidates=[]
    for j,i in enumerate(dev):
        public,private=ds[i];parts=private['episode_id'].split(':');obj=objects[parts[0]][parts[1]];category=obj.get('object_type','Unknown')
        bound=public['world_bounds'].numpy();target=bound[:2]+private['target_xy'].numpy()*(bound[2:]-bound[:2]);contour=private['target_polygon']
        result=dict(index=i,episode_id=private['episode_id'],category=category,positive_patches=0,zero_area_positives=0,patch_occupancy=[],point_polygon_distance=None)
        polygon=Polygon(contour) if contour is not None else None
        if polygon is not None and polygon.is_valid and polygon.area>0:
            result.update(polygon_area_m2=float(polygon.area),point_polygon_distance=float(polygon.distance(Point(target))))
            for fp,valid in zip(public['footprints_world'].numpy(),public['patch_valid'].numpy()):
                right=(fp[1]-fp[0])/24;down=(fp[3]-fp[0])/24;radius=.5*np.linalg.norm([np.linalg.norm(right),np.linalg.norm(down)])
                for k in np.flatnonzero(valid):
                    y,x=divmod(int(k),24);center=fp[0]+(x+.5)*right+(y+.5)*down
                    if polygon.distance(Point(center))>radius:continue
                    top=fp[0]+x*right+y*down;patch=Polygon([top,top+right,top+right+down,top+down]);occupancy=polygon.intersection(patch).area/patch.area
                    result['positive_patches']+=1;result['zero_area_positives']+=int(occupancy<1e-8);result['patch_occupancy'].append(float(occupancy))
            if result['positive_patches'] and len(candidates)<4 and category not in [x[1] for x in candidates]:candidates.append((i,category))
        rows.append(result)
        if (j+1)%100==0:print(json.dumps(dict(stage='polygon',completed=j+1,expected=len(dev))),flush=True)
    geom,_=_load_base(a.source_v2,'cuda');model=UnifiedPGVBV3(geom).cuda().eval();model.load_state_dict(torch.load(a.run/'train/last.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad];gradient=[]
    for index,category in candidates:
        kw,labels=_collate([ds[index]],'cuda');out=model(**kw);loss=v3_losses(out,labels['target_xy'],labels['target_polygons'],kw['world_bounds'],labels['teacher_features'],model.teacher_head)
        row=dict(index=index,category=category,components={})
        for name in ('geo_ce','geo_polygon','fused_ce','fused_polygon','text_visual','gt_image_visual'):
            grads=torch.autograd.grad(loss[name],[p for _,p in params],retain_graph=True,allow_unused=True)
            norm={group:float(sum(float(g.square().sum()) for (n,_),g in zip(params,grads) if g is not None and (n.startswith('geometry.'))==(group=='geometry'))**.5) for group in ('geometry','visual')}
            row['components'][name]=dict(value=float(loss[name]),gradient_norm=norm,finite=all(bool(torch.isfinite(g).all()) for g in grads if g is not None))
        gradient.append(row)
    summary={}
    for category in sorted({r['category'] for r in rows}):
        rr=[r for r in rows if r['category']==category];occ=[v for r in rr for v in r['patch_occupancy']]
        summary[category]=dict(n=len(rr),unique_physical_targets=len({':'.join(r['episode_id'].split(':')[:2]) for r in rr}),no_positive=sum(not r['positive_patches'] for r in rr),
                               positive_patches=sum(r['positive_patches'] for r in rr),zero_area_positives=sum(r['zero_area_positives'] for r in rr),occupancy_quantiles=np.quantile(occ,[0,.1,.5,.9,1]).tolist() if occ else None)
    atomic_json(dict(scope='TRAIN private weak geometry only; area overlap is not pixel truth',checkpoint_sha256=file_sha256(a.run/'train/last.pt'),N=len(rows),by_category=summary,gradients=gradient,rows=rows),a.output)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
