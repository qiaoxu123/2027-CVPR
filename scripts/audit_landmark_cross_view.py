"""TRAIN Dev same-name views with correct versus deliberately wrong correspondence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from graphnav.geometry_guided_data import load_landmark_data,batch_landmarks
from graphnav.geometry_guided_visual import LandmarkVisualBelief,visual_world_beliefs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','teacher','catalog','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    rows=[r for r in load_landmark_data(a.data,a.teacher,a.catalog) if r['metadata']['subset']=='dev' and (r['labels']==1).any() and (r['labels']==0).any()]
    replacement={}
    for r in rows:
        other=[x for x in rows if x['metadata']['query_key']==r['metadata']['query_key'] and
            x['metadata']['frame']['image_sha256']!=r['metadata']['frame']['image_sha256']]
        other.sort(key=lambda x:hashlib.sha256(('0:'+r['metadata']['sample_key']+':'+x['metadata']['sample_key']).encode()).hexdigest())
        if other:replacement[r['metadata']['sample_key']]=other[0]
    selected=[r for r in rows if r['metadata']['sample_key'] in replacement]
    native=LandmarkVisualBelief().cuda().eval().requires_grad_(False)
    trained=LandmarkVisualBelief().cuda().eval().requires_grad_(False);trained.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True)['model'])
    output={}
    with torch.inference_mode():
        for name,net,wrong in [('native_correct',native,False),('trained_correct',trained,False),('trained_wrong_view_correspondence',trained,True)]:
            result=[];groups=defaultdict(list)
            for start in range(0,len(selected),16):
                part=selected[start:start+16]
                source=[replacement[r['metadata']['sample_key']] for r in part] if wrong else None
                b=batch_landmarks(part,control='wrong_rgb' if wrong else 'real',replacement=source)
                score=net(image_features=b['image_features'],name_text=b['name_text'],type_text=b['type_text'],valid=b['valid'])
                fields=visual_world_beliefs(b['geometry_logits'],b['local_uv'],score,b['valid'],b['projection_supported'])
                top=fields['raw'].argmax(-1).cpu();local=score.masked_fill(~b['valid'],float('-inf')).argmax(-1).cpu()
                for i,r in enumerate(part):
                    ne=float(r['region_distance'][top[i]]);success=ne<=20
                    result.append(dict(sample_key=r['metadata']['sample_key'],query_key=r['metadata']['query_key'],ne=ne,success=success,
                        weak_top1=bool(r['labels'][local[i]]==1),source_frame=source[i]['metadata']['frame_key'] if wrong else r['metadata']['frame_key']))
                    groups[r['metadata']['query_key']].append(success)
            output[name]=dict(samples=len(result),named_groups=len(groups),raw_correct=sum(r['success'] for r in result),
                raw_ne=float(np.mean([r['ne'] for r in result])),weak_top1=sum(r['weak_top1'] for r in result),
                all_sampled_views_correct_names=sum(all(v) for v in groups.values()),
                mixed_success_names=sum(any(v) and not all(v) for v in groups.values()),rows=result)
    result=dict(results=output,same_destination_samples=True,selection_uses_private_target_gt=False,
        interpretation='Same landmark name/RGB source family, different real sampled XY/height view; preserve destination footprint/pose/validity/geometry. Offline wrong-correspondence control, not legal navigation input.',
        limits='Orthophoto shift/crop/scale changes at fixed yaw, not real perspective viewpoint validation; public polygon weak labels, no human instance gold.')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:{n:v for n,v in values.items() if n!='rows'} for k,values in output.items()},indent=2))


if __name__=='__main__':main()
