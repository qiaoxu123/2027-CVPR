"""Verify local Revised CityNav inputs and disclose historical binding differences."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import rasterio
from graphnav.checkpointing import file_sha256
from graphnav.semantic_geometry import valid_surface_polygon


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sbf-source',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--prepared-static',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();sys.path.insert(0,args.sbf_source)
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.coordinates import MapTransform
    from sbfnav.mapdata import GROUND_LEVEL
    root=Path(args.data_root);output=Path(args.output)
    if output.exists():raise ValueError('refusing overwrite')
    catalog=CityReferCatalog(root/'cityrefer');splits={};all_maps=set();bound_examples=[]
    for split in ['train_seen','val_seen','val_unseen']:
        data=CityNavDataset(root,split)
        prepared={}
        for line in (Path(args.prepared_static)/f'graphnav_{split}.jsonl').open():
            r=json.loads(line);prepared[(r['map_name'],r['instruction'])]=r
        counts=Counter();mismatches=[];seen=set()
        for record in data:
            key=(record.map_name,record.instruction);all_maps.add(record.map_name)
            counts['raw_episodes']+=1;counts['repeated_map_text_episode']+=int(key in seen);seen.add(key)
            counts['finite_trajectory']+=int(np.isfinite(record.trajectory).all())
            counts['target_equals_source_object_position']+=int(np.allclose(record.target_position,
                catalog.objects[record.map_name][record.object_id]['position'],atol=1e-6))
            counts['initial_height_below_ground_plus10']+=int(record.start_pose[2]<GROUND_LEVEL[record.map_name]+10)
            counts['initial_height_above_ground_plus100']+=int(record.start_pose[2]>GROUND_LEVEL[record.map_name]+100)
            references=catalog.referenced_landmarks(record)
            counts['invalid_supplied_reference_polygons']+=sum(not valid_surface_polygon(lm.contour) for lm in references)
            if key not in prepared:counts['not_in_static_query_preparation']+=1;continue
            old=prepared[key];old_ids={str(lm.get('id')) for lm in old['landmarks']};new_ids={str(lm.object_id) for lm in references}
            counts['binding_set_different_from_static_preparation']+=int(old_ids!=new_ids)
            if old_ids!=new_ids and len(mismatches)<30:
                mismatches.append(dict(map_name=record.map_name,instruction=record.instruction,
                    static_reference_ids=sorted(old_ids),sbf_reference_ids=sorted(new_ids),
                    source_reference_names=catalog.referenced_landmark_names(record)))
            t=MapTransform.from_raster(root/'rgbd'/f'{record.map_name}.tif') if record.map_name not in {x['map_name'] for x in bound_examples} else None
            if t is not None:
                bounds=np.array([t.x_min,t.y_min,t.x_max,t.y_max]);old_bounds=np.array(old['world_bounds'])
                probe=np.array([[t.x_min,t.y_min],[t.x_max,t.y_max],[(t.x_min+t.x_max)/2,(t.y_min+t.y_max)/2]])
                bound_examples.append(dict(map_name=record.map_name,maximum_prior_bound_difference_m=float(np.abs(bounds-old_bounds).max()),
                    sbf_world_roundtrip_max_error_m=float(np.abs(probe-t.normalized_to_world(t.world_to_normalized(probe))).max()),
                    tif_world_bounds=bounds.tolist(),graphnav_axis='x East/y North, rectangle normalized',sbf_axis='north-up image row/col, square padding'))
        splits[split]=dict(counts=counts,unique_map_text=len(seen),maps=sorted(data.map_names),
            annotation_sha256=file_sha256(data.annotation_path),binding_difference_examples=mismatches)
    rasters={}
    for name in sorted(all_maps):
        with rasterio.open(root/'rgbd'/f'{name}.tif') as raster:
            with Image.open(root/'rgbd'/f'{name}.png') as image:
                rasters[name]=dict(rgb_size=list(image.size),height_size=[raster.width,raster.height],
                    identical_dimensions=image.size==(raster.width,raster.height),ground_level_m=GROUND_LEVEL[name],
                    bounds=list(raster.bounds),height_sha256=file_sha256(root/'rgbd'/f'{name}.tif'),
                    rgb_sha256=file_sha256(root/'rgbd'/f'{name}.png'))
    report=dict(splits=splits,rasters=rasters,map_coordinates=bound_examples,
        object_sha256=file_sha256(root/'cityrefer/objects.json'),processed_sha256=file_sha256(root/'cityrefer/processed_descriptions.json'),
        common_binding='Same immutable objects/processed references supplied to both. SBF keeps largest homonymous polygon; GraphNav retains source fragments up to its recorded cap. These native preprocessing differences are disclosed, not silently attributed to architecture.',
        warning='Invalid source polygons and illegal initial altitudes are retained in benchmark episodes, never silently filtered. Robust handling/quality flags are model-side; episode denominators remain unchanged.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v['counts'] for k,v in splits.items()}))


if __name__=='__main__':main()
