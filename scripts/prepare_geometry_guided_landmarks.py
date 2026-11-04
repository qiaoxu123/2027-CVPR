"""TRAIN-only named-landmark observation audit; no original target labels read."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
import shapely
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.landmark_alias import canonical_name
from graphnav.landmark_region_supervision import named_union,weak_region_labels,public_type
from graphnav.public_reference_supervision import patch_world_windows
from scripts.analyze_instruction_landmark_association import PublicNames,iter_train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','dataset-profile','catalog','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('additional-cache','additional-observations','additional-filter'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();root=a.root
    if a.output.exists():raise FileExistsError(a.output)
    profile=json.loads(a.dataset_profile.read_text());dataset=Path(profile['dataset_directory'])
    if file_sha256(dataset/'manifest.json')!=profile['dataset_manifest_sha256']:raise ValueError('filtered dataset changed')
    allowed={(f"{r['area']}_block_{r['block']}",r['descriptions'][0]) for _,r in iter_train(Path(profile['processed_all_annotations']))}
    catalog=json.loads(a.catalog.read_text());names=PublicNames(catalog)
    groups={}
    for mp,objects in catalog.items():
        grouped=defaultdict(list)
        for oid,obj in objects.items():
            if obj.get('name'):grouped[canonical_name(obj['name'])].append((oid,obj))
        groups[mp]=grouped
    sources=[('old','chain_train400_features','fixed_yaw_train400_s1','public_deployment_filter.json'),
             ('new','scoped_between_train294_features_v1','scoped_between_train294_observations_v1','scoped_between_train294_public_filter_v1.json')]
    if a.additional_cache:
        if not a.additional_observations or not a.additional_filter:raise ValueError('Complete additional source required')
        sources.append(('public_sampled',a.additional_cache,a.additional_observations,a.additional_filter))
    rows=[];label_cache={};queries={};excluded=Counter();source_manifests={}
    for tag,cache,obs,filter_name in sources:
        cache=root/cache;obs=root/obs
        manifest=json.loads((cache/'manifest.json').read_text())
        if manifest['split']!='train_seen' or not manifest['complete'] or manifest['camera_yaw_rad']!=0:
            raise ValueError('Existing complete TRAIN fixed-yaw cache required')
        source_manifests[str(cache/'manifest.json')]=file_sha256(cache/'manifest.json')
        frame_lookup={r['frame_key']:r for r in map(json.loads,(obs/'public_frames.jsonl').read_text().splitlines())}
        fit_frames=set(json.loads((root/filter_name).read_text())['frame_keys'])
        for ci,chunk in enumerate(manifest['chunks']):
            path=cache/chunk['file']
            if file_sha256(path)!=chunk['sha256']:raise ValueError('feature chunk changed')
            for ri,record in enumerate(torch.load(path,weights_only=True)['records']):
                subset=record['subset'];frame=frame_lookup[record['frame_key']];mp=record['map_name']
                if subset=='fit' and record['frame_key'] not in fit_frames:continue
                if (mp,frame['instruction']) not in allowed:
                    excluded['source_instruction_removed_from_associated_dataset']+=1;continue
                if subset not in {'fit','dev','confirm'}:raise ValueError('Unexpected partition')
                windows=patch_world_windows(frame['footprint_world'])[:,0]
                supported=record['local_valid'].numpy()&(record['quality'][:,0].numpy()>=.9)
                linked={name for m in names.match(mp,frame['instruction']) for name in m['public_names']}
                if 'requested_public_name' in frame:
                    requested=frame['requested_public_name']
                    if requested not in linked:raise ValueError('Sampled public name absent from original retained instruction')
                    linked={requested}
                unique_names={}
                for name in sorted(linked):unique_names.setdefault(canonical_name(name),name)
                excluded['duplicate_canonical_name_variants']+=len(linked)-len(unique_names)
                for name in unique_names.values():
                    group=groups[mp][canonical_name(name)];key=mp+'::'+canonical_name(name)
                    typ=public_type([o['object_type'] for _,o in group])
                    if typ is None:
                        excluded['mixed_or_unsupported_public_type']+=1;continue
                    try:region=named_union([o['contour'] for _,o in group])
                    except ValueError:
                        excluded['invalid_public_polygon']+=1;continue
                    label,coverage=weak_region_labels(windows,region,supported)
                    positive=int((label==1).sum());negative=int((label==0).sum())
                    sample_key=record['frame_key']+'::'+canonical_name(name)
                    if sample_key in label_cache:raise ValueError('duplicate frame/name sample')
                    label_cache[sample_key]=dict(labels=torch.from_numpy(label),coverage=torch.from_numpy(coverage))
                    queries[key]=dict(map_name=mp,name=name,type=typ,public_ids=[oid for oid,_ in group],
                        public_polygons=[o['contour'] for _,o in group],instruction='Navigate to '+name+'.')
                    rows.append(dict(sample_key=sample_key,frame_key=record['frame_key'],task_key=record['task_key'],
                        query_key=key,map_name=mp,subset=subset,public_type=typ,cache=str(cache),chunk=chunk['file'],row=ri,
                        observations=str(obs),frame=frame,positive_windows=positive,negative_windows=negative,
                        train_region_eligible=positive>0 and negative>0,
                        geometric_support=positive>0,visibility='unknown_geometry_proxy_only',human_pixel_gt=False))
    by_subset={s:dict(samples=sum(r['subset']==s for r in rows),
        eligible=sum(r['subset']==s and r['train_region_eligible'] for r in rows),
        eligible_names=len({r['query_key'] for r in rows if r['subset']==s and r['train_region_eligible']}),
        eligible_frames=len({r['frame_key'] for r in rows if r['subset']==s and r['train_region_eligible']}),
        type_counts=dict(Counter(r['public_type'] for r in rows if r['subset']==s and r['train_region_eligible']))) for s in ('fit','dev','confirm')}
    sets={s:{r['query_key'] for r in rows if r['subset']==s} for s in by_subset}
    if any(sets[x]&sets[y] for x,y in [('fit','dev'),('fit','confirm'),('dev','confirm')]):raise ValueError('Map/name leakage')
    a.output.mkdir(parents=True)
    atomic_json(rows,a.output/'samples_public.json');atomic_json(queries,a.output/'queries_public.json')
    atomic_save(label_cache,a.output/'region_supervision.pt')
    atomic_json(dict(complete=True,by_subset=by_subset,named_queries=len(queries),excluded=dict(excluded),
        used_filtered_dataset=True,dataset_manifest_sha256=profile['dataset_manifest_sha256'],
        source_manifests=source_manifests,catalog_sha256=file_sha256(a.catalog),
        samples_sha256=file_sha256(a.output/'samples_public.json'),queries_sha256=file_sha256(a.output/'queries_public.json'),
        labels_sha256=file_sha256(a.output/'region_supervision.pt'),producer_sha256=file_sha256(__file__),
        camera_yaw_rad=0.,native_image_size=384,original_target_coordinates_and_ids_read=False,
        caveat='Public polygon projection is weak region supervision, not verified pixel identity or visibility; eligible names separated by original map partitions.'),a.output/'manifest.json')
    print(json.dumps(by_subset,indent=2),flush=True)


if __name__=='__main__':main()
