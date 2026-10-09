"""Real frozen P2 encoder on two causally generated current observations."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.observed_landmarks import ObservedLandmarkEncoder
from graphnav.checkpointing import file_sha256,provenance
from scripts.prepare_landmark_views import objects_by_map


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','episodes','objects','rgbd','sbf-source','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('refusing current-observation smoke overwrite')
    out.mkdir(parents=True);sys.path.insert(0,args.sbf_source)
    from sbfnav.mapdata import GROUND_LEVEL
    source=provenance();episodes=json.loads(Path(args.episodes).read_text())
    episode=next(row for row in episodes if 30<=row['start_pose'][2]-GROUND_LEVEL[row['episode_id'][0]]<=100)
    map_name=episode['episode_id'][0];pose=np.asarray(episode['start_pose'],dtype=float)
    renderer=CityNavObservedViewRenderer(args.rgbd,out/'observations',image_size=384)
    encoder=ObservedLandmarkEncoder(args.checkpoint,out/'actual_pixel_cache');encoder.start_episode()
    objects=objects_by_map(json.loads(Path(args.objects).read_text()));rows=[];start=time.perf_counter()
    for step in range(2):
        current=pose.copy();current[0]+=10*step
        observation=renderer.current_rgbd(map_name,current,GROUND_LEVEL[map_name])
        rois,vectors=encoder.observe(observation,objects[map_name],step=step)
        if not len(rois):raise ValueError('no trusted current-view landmark; smoke cannot silently pass')
        assert vectors.shape==(len(rois),256) and torch.isfinite(vectors).all()
        rows.append(dict(step=step,map_name=map_name,pose=current.tolist(),landmarks=len(rois),
            categories=sorted(set(roi.object_type for roi in rois)),confidence_mean=float(np.mean([roi.confidence for roi in rois])),
            vector_shape=list(vectors.shape)))
    before=encoder.backbone.encoded_images;encoder.vectors.clear()
    repeated,reloaded=encoder.observe(observation,objects[map_name],step=2)
    if encoder.backbone.encoded_images!=before or not torch.allclose(vectors,reloaded,atol=1e-4):
        raise ValueError('current-pixel persistent vectors must avoid repeated expensive encoding')
    result=dict(provenance=source,checkpoint_sha256=file_sha256(args.checkpoint),episode_manifest_sha256=file_sha256(args.episodes),
        rows=rows,history=encoder.history,persistent_vector_reuse=True,seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        input_contract='Current384 RGB/vertical-depth proxy, public polygons and current pose only. No destination/target ID/expert future.',
        note='Real interface smoke only; registration remains weak, no navigation gain.')
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
