"""TRAIN weak-label RGB review atlas; projections are not pixel gold."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_projection import project_landmark,draw_projected_roi,make_roi_image,world_to_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','audit','catalog','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--category',action='append',choices=('Car','Building','Parking','Ground'))
    p.add_argument('--per-category',type=int,default=3)
    p.add_argument('--exclude-selection',type=Path)
    a=p.parse_args()
    if not 1<=a.per_category<=48:raise ValueError('bounded review batch required')
    excluded={x['task_key'] for x in json.loads(a.exclude_selection.read_text())} if a.exclude_selection else set()
    if a.output.exists():raise FileExistsError(a.output)
    rows=json.loads((a.audit/'records.json').read_text());labels=torch.load(a.audit/'loss_labels_private.pt',weights_only=True)
    chosen=[]
    for category in (a.category or ('Car','Building','Parking','Ground')):
        rr=[x for x in rows if x['category']==category and x['counts']['frames_with_patch_pos_and_neg'] and x['task_key'] not in excluded]
        rr.sort(key=lambda x:hashlib.sha256(('0:'+x['frame_key']).encode()).hexdigest());seen=set()
        for x in rr:
            if x['task_key'] not in seen:seen.add(x['task_key']);chosen.append(x)
            if len(seen)==a.per_category:break
    objects=json.loads(a.catalog.read_text());lookup={};truth={};features={}
    for cache,obs in [('chain_train400_features','fixed_yaw_train400_s1'),('scoped_between_train294_features_v1','scoped_between_train294_observations_v1')]:
        wanted={x['frame_key'] for x in chosen}
        for f in map(json.loads,(a.root/obs/'public_frames.jsonl').read_text().splitlines()):
            if f['frame_key'] in wanted:lookup[f['frame_key']]=(f,a.root/obs)
        for f in map(json.loads,(a.root/obs/'labels_private.jsonl').read_text().splitlines()):
            if f.get('frame_key') in wanted:truth[f['frame_key']]=f
        meta=json.loads((a.root/cache/'manifest.json').read_text())
        for c in meta['chunks']:
            for f in torch.load(a.root/cache/c['file'],weights_only=True)['records']:
                if f['frame_key'] in wanted:features[f['frame_key']]=f
    a.output.mkdir(parents=True);page=['<meta charset="utf-8"><h1>TRAIN目标patch弱监督：真实RGB核验队列</h1><p>绿色是Contour几何投影，不是真实像素金标准。紫色点是保守几何正patch；橙色点是20m内但距离目标Contour外扩2m之外的点。右侧仅当前RGB裁剪插值，不增加原生细节。</p>']
    for x in chosen:
        key=x['frame_key'];frame,obs=lookup[key];private=truth[key];mp,oid=private['physical_group'];obj=objects[mp][str(oid)];image=obs/frame['image'];assert file_sha256(image)==frame['image_sha256']
        bgr=cv2.imread(str(image));roi=project_landmark({**obj,'id':str(oid)},frame['footprint_world'],bgr.shape,min_visible=0,min_pixels=1)
        if roi is None:raise ValueError('positive weak patch without projectable target polygon')
        overlay=draw_projected_roi(bgr,roi);f=features[key];xy=world_to_frame(f['candidate_world'][784:].numpy(),frame['footprint_world'],bgr.shape)
        pos=labels[key]['target_patch_geometry'].numpy()==1
        for i in np.flatnonzero(pos):cv2.circle(overlay,tuple(np.rint(xy[i]).astype(int)),3,(255,0,255),-1)
        # The orange proxy uses cached world distances and a private buffer only for review.
        import shapely
        from shapely.geometry import Polygon
        world=f['candidate_world'][784:].numpy();distance=np.linalg.norm(world-np.asarray(private['target_world'][:2]),axis=-1)
        outside=~shapely.covers(Polygon(obj['contour']).buffer(2),shapely.points(world));other=(distance<=20)&f['local_valid'].numpy()&outside
        for i in np.flatnonzero(other):cv2.circle(overlay,tuple(np.rint(xy[i]).astype(int)),2,(0,165,255),-1)
        panel=np.concatenate([bgr,overlay,make_roi_image(bgr,roi,output_size=384)],axis=1)
        assert cv2.imwrite(str(a.output/(key+'.png')),panel)
        x.update(instruction=frame['instruction'],image_sha256=frame['image_sha256'],pixel_gt=False,visibility='unknown')
        page+=['<h2>'+html.escape(key+' / '+x['category'])+'</h2><p>'+html.escape(frame['instruction'])+'</p><a href="'+key+'.png"><img width="1152" src="'+key+'.png"></a>']
    atomic_json(dict(cases=len(chosen),reviewed=False,human_verified=False,category=a.category,per_category=a.per_category,private_GT_for_annotation_only=True,producer_sha256=file_sha256(__file__),source_labels_sha256=file_sha256(a.audit/'loss_labels_private.pt'),excluded_selection_sha256=file_sha256(a.exclude_selection) if a.exclude_selection else None),a.output/'manifest.json')
    atomic_json(chosen,a.output/'selection.json');(a.output/'index.html').write_text('\n'.join(page))
    print('saved',len(chosen),'actual current RGB review cases',flush=True)

if __name__=='__main__':main()
