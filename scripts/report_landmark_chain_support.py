"""Post-hoc geometric support audit; no detection/visibility ground-truth claims."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graphnav.landmark_projection import world_to_frame
from graphnav.checkpointing import atomic_json, file_sha256


def audit(root, cache):
    frames={r['frame_key']:r for r in map(json.loads,(root/'public_frames.jsonl').read_text().splitlines())}
    labels={r['frame_key']:r for r in map(json.loads,(root/'labels_private.jsonl').read_text().splitlines()) if r['status']=='captured'}
    manifest=json.loads((cache/'manifest.json').read_text())
    rows=[]
    for chunk in manifest['chunks']:
        path=cache/chunk['file']
        if file_sha256(path)!=chunk['sha256']:
            raise ValueError('cache checksum mismatch')
        for r in torch.load(path,map_location='cpu',weights_only=True)['records']:
            f=frames[r['frame_key']]
            gt=labels[r['frame_key']]['target_world'][:2]
            pixel=world_to_frame(np.array([gt]),f['footprint_world'],(384,384,3))[0]
            inside=bool(((pixel>=0)&(pixel<=383)).all()); supported=False
            if inside:
                with np.load(root/f['depth']) as depth:
                    x,y=np.rint(pixel).astype(int)
                    supported=bool(depth['depth_valid'][y,x])
            world=r['grid_world'][r['roi_ids']].numpy()
            near=bool(len(world) and np.linalg.norm(world-np.array(gt),axis=-1).min()<=20.)
            rows.append(dict(frame_key=r['frame_key'],task_key=r['task_key'],category=labels[r['frame_key']]['category'],
                             target_point_in_footprint=inside,target_point_depth_supported=supported,
                             supported_cells=len(r['roi_ids']),supported_cell_within20m=near))
    return dict(frames=len(rows), physical_targets=len({r['task_key'] for r in rows}),
                target_point_in_footprint=sum(r['target_point_in_footprint'] for r in rows),
                target_point_depth_supported=sum(r['target_point_depth_supported'] for r in rows),
                no_valid_roi_frames=sum(not r['supported_cells'] for r in rows),
                depth_supported_but_no_roi=sum(r['target_point_depth_supported'] and not r['supported_cells'] for r in rows),
                supported_cell_within20m=sum(r['supported_cell_within20m'] for r in rows), rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('observations','legacy-cache','small-cache','output'):
        p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior audit')
    left=audit(a.observations,a.legacy_cache);right=audit(a.observations,a.small_cache)
    if {r['frame_key'] for r in left['rows']}!={r['frame_key'] for r in right['rows']}:
        raise ValueError('paired sample sets differ')
    atomic_json(dict(scope='post-hoc TRAIN geometry/depth proxies, NOT verified instance matching or visibility',
                     legacy=left,small=right,observation_manifest_sha256=file_sha256(a.observations/'manifest.json'),
                     legacy_features_manifest_sha256=file_sha256(a.legacy_cache/'manifest.json'),
                     small_features_manifest_sha256=file_sha256(a.small_cache/'manifest.json')),a.output)
    print(json.dumps({name:{k:v for k,v in values.items() if k!='rows'} for name,values in [('legacy',left),('small',right)]},indent=2))


if __name__=='__main__':main()
