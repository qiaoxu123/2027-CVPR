"""Offline actual human-observation audit; target geometry is diagnostic only.

No model, policy or training data is produced. Cameras are chosen by seeded
map-balanced episode sampling and fixed human steps, never by target location.
The height intervention uses public pose only. RGB crops come from these views.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import random
import sys

import cv2
import numpy as np

from graphnav.checkpointing import file_sha256, provenance
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.observed_scales import extract_observed_scale
from graphnav.landmark_projection import world_to_frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('sbf-source', 'data-root', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--split', choices=['train_seen', 'val_seen'], default='val_seen')
    p.add_argument('--episodes-per-map', type=int, default=2)
    p.add_argument('--steps', type=int, nargs='+', default=[0, 8, 16])
    p.add_argument('--seed', type=int, default=20261010)
    p.add_argument('--visualizations', type=int, default=36)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError('refusing audit overwrite')
    sys.path.insert(0, a.sbf_source)
    from sbfnav.dataset import CityNavDataset, CityReferCatalog
    from sbfnav.mapdata import GROUND_LEVEL
    import rasterio
    dataset = CityNavDataset(a.data_root, a.split, version='processed_citynav')
    catalog = CityReferCatalog(Path(a.data_root) / 'cityrefer')
    buckets = defaultdict(list)
    for i, episode in enumerate(dataset):
        buckets[episode.map_name].append(i)
    rng = random.Random(a.seed)
    chosen = {name: rng.sample(ids, min(a.episodes_per_map, len(ids)))
              for name, ids in sorted(buckets.items())}
    out.mkdir(parents=True)
    images = out / 'comparisons'
    images.mkdir()
    renderer = CityNavObservedViewRenderer(Path(a.data_root) / 'rgbd', out / 'unused', image_size=384)
    rows, visual_rows = [], []
    for name, indices in chosen.items():
        with rasterio.open(Path(a.data_root) / 'rgbd' / (name + '.tif')) as src:
            source_gsd = float(max(abs(src.transform.a), abs(src.transform.e)))
        for index in indices:
            record = dataset[index]
            # Diagnostic geometry is used only AFTER camera selection.
            obj = catalog.target_object(record)
            contour = np.asarray(obj['contour'], dtype=np.float32)
            for step in sorted({min(s, len(record.trajectory) - 1) for s in a.steps}):
                pose = record.trajectory[step].copy()
                before = renderer.current_rgbd(name, pose, GROUND_LEVEL[name])
                height = pose[2] - GROUND_LEVEL[name]
                can_descend = 10. < height <= 100.
                next_pose = pose.copy()
                if can_descend:
                    next_pose[2] -= min(10., height - 10.)
                after = renderer.current_rgbd(name, next_pose, GROUND_LEVEL[name]) if can_descend else None
                views = [('before', before, pose)]
                if after is not None:
                    views.append(('physical_descend', after, next_pose))
                crop_panels = []
                for view, obs, xyz in views:
                    px = world_to_frame(contour, obs['footprint_world'], obs['bgr'].shape)
                    target_px = world_to_frame(np.asarray(record.target_position[:2])[None],
                                               obs['footprint_world'], obs['bgr'].shape)[0]
                    in_fov = bool(((target_px >= 0) & (target_px <= 383)).all())
                    support = ((obs['bgr'].max(-1) > 0) & obs['depth_valid'])
                    mpp = 2. * (xyz[2] - GROUND_LEVEL[name]) / 383.
                    for extent in (32., 64., 128.):
                        roi = extract_observed_scale(obs, record.target_position[:2], extent_m=extent)
                        row = dict(map_name=name, episode_index=index, observed_step=step,
                                   observation_cutoff_step=step, view=view, extent_m=extent,
                                   category=obj.get('object_type', 'Unknown'), instruction=record.instruction,
                                   height_m=float(xyz[2]-GROUND_LEVEL[name]), source_gsd_m=source_gsd,
                                   view_m_per_native_pixel=mpp, below_source_sampling_floor=mpp < source_gsd,
                                   object_bbox_width_px=float(np.ptp(px[:, 0])),
                                   object_bbox_height_px=float(np.ptp(px[:, 1])),
                                   geometric_target_center_in_fov=in_fov,
                                   frame_rgb_depth_support=float(support.mean()),
                                   roi_available=roi is not None,
                                   roi_native_span_px=roi.native_span_px if roi else None,
                                   roi_valid_fraction=roi.valid_fraction if roi else None,
                                   roi_resampling_gain=384./roi.native_span_px if roi else None,
                                   roi_laplacian_variance=float(cv2.Laplacian(
                                       cv2.cvtColor(roi.bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if roi else None)
                        rows.append(row)
                        # Digital ROI is explicitly labeled; sharper output is not new detail.
                        if extent == 32.:
                            panel = roi.bgr.copy() if roi else np.zeros((384,384,3),np.uint8)
                            cv2.putText(panel, '32m digital ROI' if roi else 'ROI unavailable',
                                        (8,22), cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
                            crop_panels.append(panel)
                if len(visual_rows) < a.visualizations:
                    left = before['bgr'].copy()
                    cv2.putText(left, f'BEFORE h={height:.1f}m / native384', (8,22),
                                cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                    right = after['bgr'].copy() if after is not None else np.zeros_like(left)
                    cv2.putText(right, f'NEW OBS h={next_pose[2]-GROUND_LEVEL[name]:.1f}m' if can_descend
                                else 'No legal descent', (8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                    second = crop_panels[-1] if len(crop_panels)>1 else np.zeros_like(left)
                    montage=np.concatenate((np.concatenate((left,right),1),
                                            np.concatenate((crop_panels[0],second),1)),0)
                    label=f'{len(visual_rows):03d}_{name}_ep{index}_step{step}.png'
                    if not cv2.imwrite(str(images/label),montage):raise IOError(label)
                    visual_rows.append(dict(file=label,map_name=name,index=index,observed_step=step,
                                            instruction=record.instruction,category=obj.get('object_type'),
                                            blinded_roof_color=None,blinded_shape=None,blinded_vehicle=None,
                                            before_attribute_readable=None,after_attribute_readable=None))
        print(json.dumps(dict(map=name,records=len(rows))),flush=True)
    with (out/'measurements.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'visual_review_template.json').write_text(json.dumps(visual_rows,indent=2))
    summary=dict(args=vars(a),provenance=provenance(),source_script_sha256=file_sha256(__file__),
                 source_sha256=file_sha256(dataset.annotation_path),
                 objects_sha256=file_sha256(Path(a.data_root)/'cityrefer/objects.json'),
                 selected_episode_indices=chosen,records=len(rows),visualizations=len(visual_rows),
                 available_rois=sum(r['roi_available'] for r in rows),
                 annotation_status='Human/blinded attribute correctness is UNANNOTATED, not an accuracy measurement.',
                 constraints='Independent actual human observations; fixed-step selection without target. Descend is public-pose one-step intervention, not a completed navigation rollout. GT object contour/target coordinate only define offline diagnostic ROI/labels, never a policy input. ROI output upsampling/Laplacian variance is not evidence of additional physical detail or semantic recognition. Nadir orthophoto, no new 3D side view.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
