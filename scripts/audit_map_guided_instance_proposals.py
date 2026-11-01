"""Evaluation-only registered-instance oracle for original frozen900 proposals."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import torch
from torchvision.ops import box_iou
from graphnav.checkpointing import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['records','catalog','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise ValueError('Use new output')
    records=[json.loads(line) for line in Path(a.records).read_text().splitlines()]
    catalog=json.loads(Path(a.catalog).read_text());counts=defaultdict(lambda:dict(states=0,all900=0,valid900=0,top15=0))
    for r in records:
        if not r['instance']['native']['eligible']:continue
        frame=torch.load(r['raw_frame'],weights_only=True)
        boxes=frame['all_boxes_cxcywh'].float();corners=torch.cat([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],-1).clamp(0,1)
        height,width=frame['depth_valid'].shape
        gt=torch.tensor(r['target_eval_bbox']).float()[None]/torch.tensor([width,height,width,height])
        iou=box_iou(corners,gt)[:,0];quality=frame['all_valid_pixel_fraction']>=.6
        kind=catalog[r['map']][r['official_instance_id']]['object_type']
        for label in ['all',kind]:
            c=counts[label];c['states']+=1;c['all900']+=int((iou>=.5).any());c['valid900']+=int(((iou>=.5)&quality).any());c['top15']+=int(r['instance']['native']['top15'])
    atomic_json(dict(protocol='official registered bbox IoU>=0.5; read-only subset; never selection by GT',counts=dict(counts)),Path(a.output))


if __name__=='__main__':main()
