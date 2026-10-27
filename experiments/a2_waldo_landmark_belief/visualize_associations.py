"""Deterministic TRAIN label audit panels, never network inputs."""
import hashlib
import cv2
import numpy as np
import torch
from .paths import bootstrap,ROOT,RGBD,save
bootstrap()
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import world_to_frame
from sbfnav.mapdata import GROUND_LEVEL


def main():
    cache=ROOT/'runs/a2_waldo_landmark_20261027/cache_smoke_car_anchor_position_v2'
    pools={True:{},False:{}}
    for path in sorted(cache.glob('train_*.pt')):
        row=torch.load(path,map_location='cpu',weights_only=False)
        if not row['public']['network']['link_valid'].any():
            continue
        for i,detail in enumerate(row['private']['association']):
            if detail['method']!='unique_coarse_car_anchor' or not detail['supervision_allowed']:
                continue
            positive=bool(row['labels']['ranking'][i]>0)
            key=(row['target_instance'][0],detail['object_id'])
            pools[positive].setdefault(key,(path,i))
    selected=[]
    for positive in (True,False):
        ordered=sorted(pools[positive].items(),key=lambda p:hashlib.sha256(str(p[0]).encode()).digest())[:12]
        selected.extend((positive,key,*value) for key,value in ordered)
    renderer=CityNavObservedViewRenderer(RGBD,ROOT/'runs/unused',image_size=384)
    tiles=[];cases=[]
    for positive,key,path,i in selected:
        row=torch.load(path,map_location='cpu',weights_only=False)
        detail=row['private']['association'][i];candidate=row['meta']['candidates'][i]
        image=renderer.current_rgbd(key[0],np.asarray(row['meta']['pose']),GROUND_LEVEL[key[0]])['bgr']
        pixels=np.asarray(candidate['pixels'])
        center=pixels.mean(0);radius=max(float((pixels.max(0)-pixels.min(0)).max()),12.)
        lo=np.maximum(np.floor(center-radius).astype(int),0);hi=np.minimum(np.ceil(center+radius).astype(int)+1,384)
        overlay=image.copy()
        cv2.polylines(overlay,[np.rint(pixels).astype(np.int32)],True,(0,255,0),1)
        anchor=world_to_frame(np.array([detail['anchor_world_xy']],np.float32),row['meta']['footprint_world'],image.shape)[0]
        cv2.drawMarker(overlay,tuple(np.rint(anchor).astype(int)),(0,0,255),cv2.MARKER_CROSS,7,1)
        crop=overlay[lo[1]:hi[1],lo[0]:hi[0]]
        tile=np.zeros((288,256,3),np.uint8)
        tile[32:]=cv2.resize(crop,(256,256),interpolation=cv2.INTER_NEAREST)
        label=f'{"POS" if positive else "NEG"} oid={key[1]} d={detail["center_distance_m"]:.2f}m'
        cv2.putText(tile,label,(3,21),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        tiles.append(tile)
        cases.append(dict(source_row=str(path),candidate_index=i,map_name=key[0],associated_id=key[1],
                          positive=positive,confidence=candidate['confidence'],**detail))
    while len(tiles)%4:
        tiles.append(np.zeros((288,256,3),np.uint8))
    sheet=np.concatenate([np.concatenate(tiles[j:j+4],axis=1) for j in range(0,len(tiles),4)],axis=0)
    out=ROOT/'reports/a2_execution_20261027/association_cases'
    cv2.imwrite(str(out/'anchor_review_position_v2.png'),sheet)
    save(dict(scope='Deterministic TRAIN-only linked-reference positive and hard-negative label audit',
              legend='green observed WALDO box; red authoritative CityRefer position; nearest-neighbour display enlargement',cases=cases),
         out/'anchor_review_position_v2.json')
    print('Saved',len(cases),'association audit tiles',flush=True)


if __name__=='__main__':
    main()
