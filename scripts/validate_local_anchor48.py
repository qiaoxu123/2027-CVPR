"""No model execution: compare local public Anchors with the prior48 diagnostics."""
import argparse,json,collections
from pathlib import Path
import cv2
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.local_landmark_anchor import local_landmark_anchor,named_public_polygons
from graphnav.observed_landmarks import current_landmark_rois


def main():
    p=argparse.ArgumentParser()
    for k in ('audit','catalog','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);cases=json.loads((a.audit/'samples.json').read_text());catalog=json.loads(a.catalog.read_text());rows=[];contacts=[]
    for case in cases:
        i=case['index'];folder=a.audit/f'case_{i:02d}';bgr=cv2.imread(str(folder/'rgb.png'));depth=cv2.imread(str(folder/'depth_valid.png'),0)>0
        assert file_sha256(folder/'rgb.png')==case['rgb_sha256']
        obs=dict(bgr=bgr,depth_valid=depth,footprint_world=np.asarray(case['footprint_world']))
        group=named_public_polygons(catalog[case['map_name']],case['public_name'],case['type']);anchor=local_landmark_anchor(group['contours'],obs)
        legacy=[dict(id=oid,name=obj['name'],object_type=obj['object_type'],contour=obj['contour']) for oid,obj in catalog[case['map_name']].items() if str(oid) in group['ids']]
        old_any=bool(current_landmark_rois(obs,legacy))
        labels=anchor['labels'];old_pgvb=bool((labels==1).sum()>=2 and (labels==0).sum()>=4)
        row=dict(index=i,type=case['type'],state=case['state'],name=case['public_name'],source_fragment_roi=case['thresholds']['current_roi90_pipeline'],
            old_any_named_roi=old_any,old_named_pgvb_selection=old_pgvb,new_pixel_anchor=anchor['pixel_usable'],new_train_eligible=anchor['train_eligible'],
            whole_fraction=anchor['whole_fraction'],view_fraction=anchor['view_fraction'],valid_pixels=anchor['stable_valid_pixels'],
            positive=anchor['positive_patches'],negative=anchor['negative_patches'],positive_only=anchor['positive_only'],
            source_rgb_sha256=case['rgb_sha256'],actual_visibility='unknown',all_named_ids=group['ids'])
        rows.append(row);out=a.output/f'case_{i:02d}';out.mkdir()
        cv2.imwrite(str(out/'local_mask.png'),anchor['valid_mask'].astype(np.uint8)*255)
        overlay=bgr.copy();overlay[anchor['valid_mask']]=(overlay[anchor['valid_mask']]*.65+np.array([0,255,0])*.35).astype(np.uint8)
        cv2.imwrite(str(out/'local_anchor.png'),overlay)
        tiny=cv2.resize(overlay,(256,256));head=np.full((45,256,3),255,np.uint8);cv2.putText(head,f'{i:02d} {case["type"]} pos{row["positive"]} neg{row["negative"]}',(4,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        cv2.putText(head,f'ROI {int(old_any)} -> Local {int(row["new_pixel_anchor"])}',(4,37),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1);contacts.append(np.vstack([head,tiny]))
    summary=dict(n=48,legacy_source_roi=sum(r['source_fragment_roi'] for r in rows),legacy_any_named_roi=sum(r['old_any_named_roi'] for r in rows),
        new_pixel_anchor=sum(r['new_pixel_anchor'] for r in rows),legacy_pgvb_named=sum(r['old_named_pgvb_selection'] for r in rows),new_train_eligible=sum(r['new_train_eligible'] for r in rows),
        positive_only=sum(r['positive_only'] and r['new_train_eligible'] for r in rows),
        by_type={t:dict(n=len(rr),old_roi=sum(r['old_any_named_roi'] for r in rr),new_pixel=sum(r['new_pixel_anchor'] for r in rr),old_train=sum(r['old_named_pgvb_selection'] for r in rr),new_train=sum(r['new_train_eligible'] for r in rr)) for t in ('Building','Road') if (rr:=[r for r in rows if r['type']==t])},
        no_model_or_training=True,pixel_truth=False)
    for page in range(4):cv2.imwrite(str(a.output/f'contact_{page}.png'),np.vstack([np.hstack(contacts[j:j+3]) for j in range(page*12,page*12+12,3)]))
    atomic_json(rows,a.output/'rows.json');atomic_json(summary,a.output/'summary.json');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
