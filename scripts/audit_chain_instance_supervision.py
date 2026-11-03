"""Prepare private weak TRAIN Car-core labels; never modify inference features."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from graphnav.training_instance_labels import instance_core_labels,conservative_car_background,instance_annotation_counts
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','sbf-source','cache','labels','fit-observation-filter','extra-cache','extra-labels','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--background-margin-m',type=float)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous annotation audit')
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    ds=CityNavDataset(a.data_root,'train_seen',version='processed_citynav');objects=json.loads((a.data_root/'cityrefer/objects.json').read_text());meta=json.loads((a.cache/'manifest.json').read_text());maps={m for m,s in meta['map_assignment'].items() if s=='fit'}
    ids={m:set() for m in maps}
    for raw in ds._raw_records:
        name=f"{raw['area']}_block_{raw['block']}";oid=int(raw['object_ids'][0])
        if name in ids and objects[name][str(oid)]['object_type']=='Car':ids[name].add(oid)
    registry={m:(sorted(v),np.array([objects[m][str(i)]['position'][:2] for i in sorted(v)],float)) for m,v in ids.items()}
    allowed=set(json.loads(a.fit_observation_filter.read_text())['frame_keys']);truth={x['frame_key']:x for path in (a.labels,a.extra_labels) for x in map(json.loads,path.read_text().splitlines())}
    counts=Counter();labels={}
    for cache,is_original in ((a.cache,True),(a.extra_cache,False)):
        manifest=json.loads((cache/'manifest.json').read_text())
        if not manifest['complete'] or manifest['split']!='train_seen':raise ValueError('complete TRAIN cache only')
        for chunk in manifest['chunks']:
            path=cache/chunk['file']
            if file_sha256(path)!=chunk['sha256']:raise ValueError('feature SHA changed')
            for row in torch.load(path,map_location='cpu',weights_only=True)['records']:
                key=row['frame_key']
                if row['subset']!='fit' or key not in truth or truth[key]['category']!='Car' or (is_original and key not in allowed):continue
                private=truth[key];name=row['map_name'];target_id=private['physical_group'][1];object_ids,xy=registry[name];world=row['candidate_world'][784:].numpy();valid=row['local_valid'].numpy()
                mask=instance_core_labels(world,valid,xy,object_ids,target_id)
                core=mask.copy()
                if a.background_margin_m is not None:
                    mask=conservative_car_background(mask,world,valid,objects[name][str(target_id)]['contour'],a.background_margin_m)
                counts.update(instance_annotation_counts(core,mask))
                labels[key]=torch.from_numpy(mask);distance=np.linalg.norm(world-np.array(private['target_world'][:2]),axis=-1)
                counts['car_frames']+=1
                for radius in (5,20):
                    near=(distance<=radius)&valid;counts[f'candidate_points_within{radius}']+=int(near.sum());counts[f'other_car_core_within{radius}']+=int((near&(core==0)).sum());counts[f'target_core_within{radius}']+=int((near&(core==1)).sum())
    a.output.mkdir(parents=True);torch.save(labels,a.output/'loss_labels_private.pt')
    result=dict(background_margin_m=a.background_margin_m,counts=dict(counts),radius_m=2.,minimum_neighbor_margin_m=.75,annotation_sha256=file_sha256(ds.annotation_path),catalog_sha256=file_sha256(a.data_root/'cityrefer/objects.json'),
        primary_feature_manifest_sha256=file_sha256(a.cache/'manifest.json'),extra_feature_manifest_sha256=file_sha256(a.extra_cache/'manifest.json'),private_labels_sha256=file_sha256(a.output/'loss_labels_private.pt'),
        restriction='Only registered object IDs appearing in TRAIN fitting annotations; no Val/Test labels and no private labels in model forward.',
        warning='Weak geometry cores, NOT verified pixel-instance labels; nearest-object ties/invalid points/far background remain unknown. Prepared for future loss only, not yet trained.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
