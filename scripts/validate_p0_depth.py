"""Add native aligned orthographic depth diagnostics to existing P0 audit cases."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import frame_to_world


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--arrays',required=True)
    args=p.parse_args()
    root=Path(args.data_root);output=Path(args.output);arrays=Path(args.arrays)
    if output.exists() or arrays.exists():raise ValueError('refusing overwrite')
    output.mkdir(parents=True);arrays.mkdir(parents=True)
    cases=json.loads(Path(args.cases).read_text())
    from sbfnav.mapdata import GROUND_LEVEL
    renderer=CityNavObservedViewRenderer(root/'rgbd',arrays,image_size=384)
    trajectories={split:json.loads((root/'processed_citynav'/f'citynav_{split}.json').read_text())
                  for split in {r['split'] for r in cases}}
    results=[]
    for case in sorted(cases,key=lambda c:(c['roi']['map_name'],c['case_id'])):
        row=case['roi'];source=trajectories[case['split']][row['source_episode_index']]
        raw=source['trajectory'][row['observed_step']]
        yaw=raw[3] if len(raw)==5 else np.arctan2(raw[4],raw[3])
        obs=renderer.current_rgbd(row['map_name'],[*raw[:3],yaw],GROUND_LEVEL[row['map_name']])
        saved=cv2.imread(row['observed_frame'])
        if not np.array_equal(saved,obs['bgr']):raise ValueError('native RGB differs from original P0')
        np.savez_compressed(arrays/f"case_{case['case_id']:02d}.npz",depth_m=obs['depth_m'],valid=obs['depth_valid'])
        mask=np.zeros((384,384),np.uint8);points=np.rint(row['contour_px']).astype(np.int32)
        cv2.fillPoly(mask,[points],1);region=mask>0
        valid_fraction=float(obs['depth_valid'][region].mean()) if region.any() else 0
        # Numeric uncertainty is an explicit 2px boundary perturbation, not measured annotation error.
        corners=np.array(row['contour_px'],np.float32)
        original=frame_to_world(corners,row['footprint_world'],(384,384))
        shifted=frame_to_world(corners+np.array([2.,0.],np.float32),row['footprint_world'],(384,384))
        uncertainty=float(np.linalg.norm(shifted-original,axis=1).mean())
        depth=cv2.applyColorMap(np.clip(obs['depth_m']/100*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
        depth[~obs['depth_valid']]=0;rgb=obs['bgr'].copy()
        cv2.polylines(rgb,[points],True,(0,255,0),1);cv2.polylines(depth,[points],True,(0,255,0),1)
        image=output/f"case_{case['case_id']:02d}.jpg";cv2.imwrite(str(image),np.concatenate([rgb,depth],axis=1))
        results.append(dict(case_id=case['case_id'],split=case['split'],map_name=row['map_name'],
            object_type=row['object_type'],native_rgb_exact_match=True,
            projected_region_valid_depth_fraction=valid_fraction,boundary_2px_world_uncertainty_m=uncertainty,
            depth_array=str((arrays/f"case_{case['case_id']:02d}.npz").resolve()),visualization=str(image)))
    by_type=defaultdict(list)
    for r in results:by_type[r['object_type']].append(r)
    report=dict(cases=results,by_type={k:dict(cases=len(v),mean_valid_depth_fraction=float(np.mean([r['projected_region_valid_depth_fraction'] for r in v])),
        mean_2px_boundary_uncertainty_m=float(np.mean([r['boundary_2px_world_uncertainty_m'] for r in v]))) for k,v in by_type.items()},
        depth_definition='Orthographic vertical-depth proxy from observed surface height; not perspective ray depth, no 3D occlusion renderer.',
        GT_access='Only sampled current teacher pose for offline P0 diagnostic; no target or later pose read by renderer.')
    (output/'depth_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['by_type']))


if __name__=='__main__':main()
