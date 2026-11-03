"""Annotation-side schema only; AI boxes never become verified pixel gold."""
import math


def validate_weak_box_record(record, image_sha256, image_shape=(384,384)):
    if record.get('image_sha256')!=image_sha256:raise ValueError('source image SHA changed')
    if record.get('human_verified') is not False or record.get('pixel_gold') is not False:
        raise ValueError('AI annotations must remain explicitly weak and nonhuman')
    if record.get('annotation_only') is not True:raise ValueError('TRAIN annotation-only record required')
    if tuple(record.get('image_shape',(384,384)))!=tuple(image_shape):raise ValueError('native annotation dimensions differ from current RGB')
    h,w=image_shape;boxes=record.get('boxes_px',[])
    if record.get('assessment')!='candidate' and boxes:raise ValueError('abstentions/groups cannot become single-instance boxes')
    if record.get('target_unit')=='vehicle_group' and boxes:raise ValueError('group region is not a single vehicle instance')
    if len(boxes)>1:raise ValueError('single-vehicle appearance proposals only in this export')
    group=record.get('group_region_bbox_px')
    if group is not None:
        if record.get('target_unit')!='vehicle_group' or record.get('target_count',0)<2 or record.get('group_region_not_instance_bbox') is not True:
            raise ValueError('group region must preserve plural target semantics')
        if len(group)!=4 or not all(math.isfinite(float(v)) for v in group):raise ValueError('finite group region required')
        if not (0<=group[0]<group[2]<=w and 0<=group[1]<group[3]<=h):raise ValueError('group region outside source RGB')
    normalized=[]
    for box in boxes:
        if len(box)!=4 or not all(math.isfinite(float(v)) for v in box):raise ValueError('finite XYXY box required')
        x0,y0,x1,y1=box
        if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('box outside source native RGB')
        normalized.append([x0/w,y0/h,x1/w,y1/h])
    if record.get('eligible_for_instance_loss') or record.get('eligible_for_goal_loss'):
        raise ValueError('unverified appearance proposals do not supervise instance identity or world Goal')
    return normalized
