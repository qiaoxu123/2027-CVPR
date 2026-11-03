"""Current RGB/depth support for public reference alternatives, never target GT."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.landmark_projection import project_landmark, make_roi_image, draw_projected_roi
from graphnav.observed_landmarks import current_landmark_rois


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('audit','observations','catalog','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    source=json.loads((a.audit/'records.json').read_text())
    catalog=json.loads(a.catalog.read_text())
    assert file_sha256(a.catalog)==json.loads((a.audit/'summary.json').read_text())['input_sha256']['catalog']
    frames={x['frame_key']:x for x in map(json.loads,(a.observations/'public_frames.jsonl').read_text().splitlines())}
    grouped={}
    for row in source:grouped.setdefault(row['frame_key'],[]).append(row)
    rows=[]
    for key,groups in grouped.items():
        frame=frames[key];path=a.observations/frame['image'];depth_path=a.observations/frame['depth']
        assert file_sha256(path)==frame['image_sha256']
        assert file_sha256(depth_path)==frame['depth_sha256']
        bgr=cv2.imread(str(path));assert bgr is not None and bgr.shape==(384,384,3)
        with np.load(depth_path) as d:valid=d['depth_valid'].copy()
        ids=sorted({m['public_id'] for r in groups for m in r['members']})
        landmarks=[dict(id=i,**{k:catalog[frame['map_name']][i][k] for k in ('name','object_type','contour')}) for i in ids]
        actual={r.object_id:r for r in current_landmark_rois(dict(bgr=bgr,depth_valid=valid,footprint_world=frame['footprint_world']),landmarks,max_rois=max(1,len(ids)))}
        for group in groups:
            for m in group['members']:
                i=m['public_id'];obj=catalog[frame['map_name']][i]
                roi=project_landmark({**obj,'id':i},frame['footprint_world'],bgr.shape,min_visible=0,min_pixels=32)
                if roi is None:continue
                mask=roi.mask.astype(bool)
                x0,y0,x1,y1=roi.bbox_xyxy
                rows.append(dict(frame_key=key,subset=group['subset'],map_name=frame['map_name'],
                    public_id=i,name=obj['name'],object_type=obj['object_type'],group_kind=group['group']['kind'],
                    original_instruction=frame['instruction'],image=frame['image'],image_sha256=frame['image_sha256'],
                    projected_fraction=roi.visible_fraction,projected_pixels=roi.visible_pixels,
                    min_bbox_side=min(x1-x0,y1-y0),rgb_support=float((bgr.max(-1)>0)[mask].mean()),
                    depth_support=float(valid[mask].mean()),accepted_current_roi=i in actual,
                    current_roi_quality=actual[i].quality if i in actual else None,
                    eligible_patch_scales=m['eligible_patch_scales'],visibility='unknown',pixel_gt=False))
    # One deterministic representative per fit-only group/status stratum, then fill
    # remaining slots by hash; no target coordinates, outcomes or hand-picked RGB.
    fit=[r for r in rows if r['subset']=='fit']
    order=sorted(fit,key=lambda r:hashlib.sha256((r['frame_key']+'::'+r['public_id']).encode()).hexdigest())
    selected=[];seen_frames=set();strata=set()
    for r in order:
        stratum=(r['group_kind'],r['accepted_current_roi'])
        if stratum in strata or r['frame_key'] in seen_frames:continue
        strata.add(stratum);seen_frames.add(r['frame_key']);selected.append(r)
    for r in order:
        if len(selected)>=12:break
        if r['frame_key'] not in seen_frames:selected.append(r);seen_frames.add(r['frame_key'])
    a.output.mkdir(parents=True);(a.output/'frames').mkdir()
    page=['<meta charset="utf-8"><title>Public reference ROI audit</title><h1>公开参照ROI：真实当前RGB</h1><p>绿色轮廓仅为公开地图几何投影；不是实例金标准。右侧为同一当前图像裁剪，插值不增加细节。未知可见性，不标注目标。</p>']
    for r in selected:
        frame=frames[r['frame_key']];bgr=cv2.imread(str(a.observations/frame['image']))
        obj=catalog[r['map_name']][r['public_id']]
        roi=project_landmark({**obj,'id':r['public_id']},frame['footprint_world'],bgr.shape,min_visible=0,min_pixels=32)
        panel=np.concatenate([bgr,draw_projected_roi(bgr,roi),make_roi_image(bgr,roi,output_size=384)],axis=1)
        name=r['frame_key']+'__'+r['public_id']+'.png'
        assert cv2.imwrite(str(a.output/'frames'/name),panel)
        page.append('<section><h2>'+html.escape(r['frame_key']+' / '+r['name']+' / '+r['object_type'])+'</h2><p>'+html.escape(r['original_instruction'])+'</p><p>'+html.escape(json.dumps({k:r[k] for k in ('group_kind','accepted_current_roi','projected_fraction','projected_pixels','rgb_support','depth_support','eligible_patch_scales')}))+'</p><a href="frames/'+name+'"><img width="1152" src="frames/'+name+'"></a></section>')
    summary=dict(private_target_labels_read=False,normal_inference_modified=False,roi_cap_disabled_for_audit=True,
        selected_fit_cases=len(selected),selection='fit-only hash order; group-kind x ROI acceptance strata; distinct frames',by_subset={})
    for subset in ('fit','dev'):
        rr=[r for r in rows if r['subset']==subset];accepted=[r for r in rr if r['accepted_current_roi']]
        summary['by_subset'][subset]=dict(projected_member_frames=len(rr),projected_distinct_frames=len({r['frame_key'] for r in rr}),
            accepted_member_frames=len(accepted),accepted_distinct_frames=len({r['frame_key'] for r in accepted}),
            accepted_group_kinds=dict(Counter(r['group_kind'] for r in accepted)),
            accepted_depth_below_half=sum(r['depth_support']<.5 for r in accepted))
    summary['input_sha256']={k:file_sha256(v) for k,v in dict(audit=a.audit/'records.json',catalog=a.catalog,frames=a.observations/'public_frames.jsonl',producer=Path(__file__)).items()}
    atomic_json(rows,a.output/'records.json');atomic_json(selected,a.output/'selected.json');atomic_json(summary,a.output/'summary.json')
    (a.output/'index.html').write_text('\n'.join(page));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
