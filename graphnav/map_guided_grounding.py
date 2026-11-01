"""No-training map-guided grounding. All inference inputs are public/current."""
import re

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon, box as rectangle

from .landmark_projection import world_to_frame, frame_to_world
from .semantic_geometry import canonical, parse_spatial_relations, prior_and_rule_fields

STOPWORDS=set('a an the is it its this that of to in on at and with has which there called'.split())
RELATION_START=re.compile(r'\b(?:north(?:east|west)?|south(?:east|west)?|east|west|near|beside|between|along|behind|next to|in front of|on|to the right|to the left)\b')


def project_polygon(contour_world,footprint,image_shape,depth_valid):
    """Geometry/depth-supported coverage, NOT semantic visibility or segmentation."""
    h,w=image_shape[:2]
    empty=dict(status='unknown',valid_roi=False,coverage=0.,depth_support=0.,pixels=0,
               contour_px=[],bbox=None,reason='invalid_polygon',roundtrip_error_m=None)
    xy=np.asarray(contour_world,np.float32)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.isfinite(xy).all():return empty
    polygon=Polygon(xy)
    if not polygon.is_valid or polygon.area<=1e-6:return empty
    try:px=world_to_frame(xy,footprint,image_shape)
    except (ValueError,cv2.error):return dict(empty,reason='invalid_footprint')
    projected=Polygon(px)
    if not projected.is_valid or projected.area<=1e-6:return empty
    clipped=projected.intersection(rectangle(0,0,w-1,h-1))
    coverage=float(clipped.area/projected.area)
    roundtrip=float(np.linalg.norm(frame_to_world(px,footprint,image_shape)-xy,axis=-1).max())
    status='outside' if coverage<=1e-6 else 'full' if coverage>=.95 else 'partial'
    mask=np.zeros((h,w),np.uint8)
    if status!='outside':cv2.fillPoly(mask,[np.rint(np.clip(px,-1e6,1e6)).astype(np.int32)],1)
    ys,xs=np.where(mask!=0);pixels=len(xs)
    support=float(np.asarray(depth_valid,bool)[mask.astype(bool)].mean()) if pixels else 0.
    bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1] if pixels else None
    valid=bool(pixels>=64 and support>=.6 and coverage>=.1 and bbox and min(bbox[2]-bbox[0],bbox[3]-bbox[1])>=4)
    return dict(status=status,valid_roi=valid,coverage=coverage,depth_support=support,pixels=pixels,
                contour_px=px.tolist(),bbox=bbox,reason='supported' if valid else 'outside' if status=='outside' else 'tiny_clipped_or_invalid_depth',
                roundtrip_error_m=roundtrip)


def public_landmarks(catalog,map_name):
    """Named public map objects only. No target ID lookup or descriptions."""
    result=[]
    for item in catalog[map_name].values():
        if not item.get('name'):continue
        result.append(dict(key=str(item['id']),name=str(item['name']),kind=str(item['object_type']),
                           geometry_m=np.asarray(item['contour'],np.float32),
                           position=np.asarray(item['position'],np.float32)))
    return sorted(result,key=lambda x:x['key'])


def target_word_mask(instruction,tokenizer,length,valid_mask):
    caption=instruction.lower().strip().rstrip('.')+'.'
    tokenized=tokenizer(caption,return_offsets_mapping=True,add_special_tokens=True)
    if len(tokenized['input_ids'])!=length:raise ValueError('Cached AerialVG text/tokenizer alignment mismatch')
    match=RELATION_START.search(caption);end=match.start() if match else len(caption)
    selected=[]
    for start,stop in tokenized['offset_mapping']:
        word=caption[start:stop].strip(' .,;!?')
        selected.append(bool(stop>start and start<end and word and word not in STOPWORDS))
    mask=torch.tensor(selected,dtype=torch.bool)&valid_mask.cpu().bool()
    # A missing appearance clause is unknown, not permission to use reference words.
    return mask


def reference_crop_bounds(image_shape,projection):
    if not projection['valid_roi']:raise ValueError('Unsupported reference crop')
    x0,y0,x1,y1=projection['bbox'];pad=round(.1*max(x1-x0,y1-y0))
    h,w=image_shape[:2]
    return max(0,x0-pad),max(0,y0-pad),min(w,x1+pad),min(h,y1+pad)


def crop_reference(bgr,projection):
    x0,y0,x1,y1=reference_crop_bounds(bgr.shape,projection)
    return bgr[y0:y1,x0:x1].copy()


def reference_caption(landmark):
    kind=landmark['kind'].casefold()
    word={'trafficroad':'road','road_surface':'road','parking':'parking lot','footpath':'path','bridge':'bridge','ground':'ground'}.get(kind,kind)
    return f'a {word} called {landmark["name"]}'


def reference_candidates(landmarks,projections,instruction):
    text=' '+canonical(instruction)+' '
    ids=[i for i,e in enumerate(landmarks) if (' '+canonical(e['name'])+' ') in text]
    # Existing world relation parser uses the full public catalog, not GT-nearest names.
    entities=[dict(e,kind='road' if e['kind'].casefold()=='verified_centerline' else e['kind'].casefold()) for e in landmarks]
    relations=parse_spatial_relations(instruction,entities)
    priority={a for r in relations for group in r.anchors for a in group}
    visible=[i for i in ids if projections[i]['valid_roi']]
    visible.sort(key=lambda i:(i not in priority,-projections[i]['coverage']*projections[i]['depth_support'],landmarks[i]['key']))
    return entities,relations,ids,visible[:2]


def geometry_scores(centers,a0_at_queries,entities,relations):
    if relations:
        _,_,rules=prior_and_rule_fields(np.asarray(centers,np.float32),entities,relations,sigma_m=25.)
        fields=torch.from_numpy(np.maximum(rules,-40)/10.)
    else:fields=torch.zeros(len(centers),0)
    prior=(a0_at_queries.float()-a0_at_queries.float().max()).clamp(-8,0)
    return prior,fields


def permute_visual_correspondence(target_features,reference_features,valid):
    """Same current frame, fixed boxes/ROIs; only feature-to-region association changes."""
    ids=valid.nonzero().flatten()
    combined=torch.cat([target_features[ids],reference_features],0)
    order=torch.arange(len(combined)-1,-1,-1,device=combined.device)
    shuffled=combined[order]
    target=target_features.clone();target[ids]=shuffled[:len(ids)]
    return target,shuffled[len(ids):],order


def score_candidates(prior,fields,relations,target_features,target_language,
                     reference_features,reference_language,reference_indices,valid):
    """Fixed constants, public geometry and actual frozen visual features only."""
    map_score=prior+fields.sum(-1)
    appearance=target_features.float()@target_language.float() if target_language is not None else torch.zeros_like(prior)
    if valid.any():appearance=(appearance-appearance[valid].max()).clamp(-4,0)
    appearance=appearance*.5
    confidence={}
    for j,index in enumerate(reference_indices):
        confidence[index]=float((reference_features[j].float()@reference_language[j].float()).clamp(-20,20).sigmoid())
    visual_fields=fields.clone()
    for j,relation in enumerate(relations):
        anchor_conf=[]
        for group in relation.anchors:
            known=[confidence[i] for i in group if i in confidence]
            if known:anchor_conf.append(max(known))
        # No current reference observation is UNKNOWN: retain map evidence.
        if anchor_conf:visual_fields[:,j]*=.5+.5*sum(anchor_conf)/len(anchor_conf)
    rgb_score=prior+visual_fields.sum(-1)+appearance
    return map_score.masked_fill(~valid,-1e4),rgb_score.masked_fill(~valid,-1e4),confidence


def registered_instance_evaluation(target_contour,footprint,image_shape,depth_valid,boxes,scores):
    """Evaluation only: official instance contour registration, not a 20m label."""
    projection=project_polygon(target_contour,footprint,image_shape,depth_valid)
    eligible=projection['valid_roi'] and projection['coverage']>=.8
    if not eligible:return dict(eligible=False,projection=projection,iou=None,top1=None,top5=None,top15=None)
    from torchvision.ops import box_iou
    h,w=image_shape[:2];gt=torch.tensor(projection['bbox'],dtype=torch.float32)[None]/torch.tensor([w,h,w,h])
    corners=torch.cat([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],-1).clamp(0,1)
    iou=box_iou(corners.float(),gt)[:,0]
    order=scores.argsort(descending=True)
    return dict(eligible=True,projection=projection,iou=iou.tolist(),
                top1=bool(iou[order[:1]].max()>=.5),top5=bool(iou[order[:5]].max()>=.5),
                top15=bool(iou[order[:15]].max()>=.5))
