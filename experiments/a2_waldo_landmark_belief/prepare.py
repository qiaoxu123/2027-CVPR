"""Resumable full train/dev extraction and independent A0-on-policy calibration.

Public perception and private labels are saved separately. Sparse causal teacher
views are used for train/dev. Calibration follows actual frozen A0 decisions.
"""
import argparse
import hashlib
from collections import Counter
import json
import time
import cv2
import numpy as np
import torch
from .paths import bootstrap,ROOT,DATA,RGBD,save,sha,verify_frozen_assets
bootstrap()
from .policy import A2Policy
from .private_labels import PrivateAssociationIndex,label_view
from .data_protocol import validate_source_manifest
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout
from graphnav.dynamic_runtime import NativeObservationBuilder
from sbfnav.dataset import CityNavDataset,CityReferCatalog
from sbfnav.planning_state import PlanningStateBuilder
from sbfnav.navigation import BoundedWaypointController
from sbfnav.mapdata import GROUND_LEVEL


OUTPUT=ROOT/'runs/a2_waldo_landmark_20261027'


def cache_signature():
    files=['data_protocol.py','mentions.py','core.py','observed_geometry.py','perception.py','policy.py','private_labels.py','paths.py','prepare.py']
    source=json.loads((OUTPUT/'source_manifest.json').read_text())
    ds=CityNavDataset(DATA,'train_seen')
    validate_source_manifest(source,ds)
    raster_paths={(root/f'{name}.{ext}').resolve()
                  for root in (RGBD,DATA/'rgbd') for name in ds.map_names for ext in ('png','tif')}
    rasters={str(path):sha(path) for path in sorted(raster_paths)}
    return dict(version=1,source_manifest_sha256=sha(OUTPUT/'source_manifest.json'),
                live_annotation_sha256=sha(ds.annotation_path),rgbd_root=str(RGBD.resolve()),
                transform_raster_root=str((DATA/'rgbd').resolve()),rgbd_sha256=rasters,
                dependencies=verify_frozen_assets(),
                public_objects_sha256=sha(DATA/'cityrefer/objects.json'),
                reference_links_sha256=sha(DATA/'cityrefer/processed_descriptions.json'),
                implementation={name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name) for name in files},
                train_dev_observations='three legal teacher quantiles .25,.75,1; causal sparse history',
                calibration_observations='frozen A0 closed-loop horizon20 original controller',
                detector=dict(confidence=.25,imgsz=1024,quantize=16,max_det=300,native_rgb=384),
                association=dict(iou=.5,margin=.15,normalized_center_distance=.5,
                                 intersection='exact shapely polygon; no raster approximation',
                                 coarse_car_anchor=dict(min_contour_vertices=8,min_extent_m=6,max_aspect=1.6,min_circularity=.7,
                                                        max_center_distance_m=1.,min_bidirectional_margin_m=1.,
                                                        min_detection_confidence=.5,requires_anchor_in_box=True,
                                                        anchor_source='finite object.position, inside annotation region',
                                                        requires_exactly_one_car_anchor_in_box_across_all_partitions=True),
                                 negatives='trusted same target category and target partition only'))


def view_counts(row):
    y=row['labels'];meta=row['meta'];private=row['private']
    positive=bool((y['ranking']>0).any())
    negative=bool((y['rank_label_mask'] & (y['ranking']<=0)).any())
    eligible=bool(row['public']['network']['link_valid'].any())
    return Counter(views=1,candidates=len(meta['candidates']),target_visible=int(private['target_visible']),
                   positive_views=int(positive),informative_views=int(positive and negative and eligible),
                   known_candidates=int(y['rank_label_mask'].sum()),
                   trusted_associations=sum(a['trusted'] for a in private['association']),
                   reference_eligible_views=int(eligible),empty_detection_views=int(not meta['candidates']),
                   capped_views=int(meta['detector_cap_reached']),detector_failures=int(meta['detector_error'] is not None),
                   supported_candidates=int(row['public']['spatial']['candidate_support'].any(-1).sum()),
                   offset_positive_views=int(y['offset_valid'].any()),
                   match_valid_views=int(y['match_valid']),
                   match_positive_views=int(y['match_valid'] and y['match']>0),
                   match_negative_views=int(y['match_valid'] and y['match']<=0),
                   reliable_box_positive_labels=int((y['rank_label_mask'] & y['offset_valid']).sum()),
                   reliable_box_negative_labels=int((y['rank_label_mask'] & ~y['offset_valid']).sum()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke-episodes',type=int,default=None)
    args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(0)
    smoke=args.smoke_episodes is not None
    directory=OUTPUT/('cache_smoke' if smoke else 'cache')
    directory.mkdir(parents=True,exist_ok=True)
    signature=cache_signature()
    signature_file=directory/'signature.json'
    if signature_file.exists() and json.loads(signature_file.read_text())!=signature:
        raise ValueError('Cache provenance changed; use a new cache version, never mix rows')
    save(signature,signature_file)
    signature_sha=sha(signature_file)
    source=json.loads((OUTPUT/'source_manifest.json').read_text())
    ds=CityNavDataset(DATA,'train_seen')
    catalog=CityReferCatalog(DATA/'cityrefer')
    index=PrivateAssociationIndex(catalog)
    builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28),
                                     RGBD,OUTPUT/'render_unused')
    policy=A2Policy(OUTPUT/'cache_query',collect=True)
    controller=BoundedWaypointController(horizontal_step_m=50.,vertical_step_m=10.,minimum_above_ground_m=10.,maximum_above_ground_m=100.)
    counters={s:Counter() for s in ('train','dev','calibration')}
    entries=[]
    started=time.perf_counter()
    for partition in ('train','dev','calibration'):
        selected=[e for e in source['entries'] if e['split']==partition]
        if smoke:
            # Avoid a misleading pilot of consecutive rephrasings of one
            # instance. Choose one query per target, across the full TRAIN map
            # population, before applying the explicitly partial smoke limit.
            unique={}
            for entry in selected:
                unique.setdefault((entry['map_name'],entry['object_id']),entry)
            selected=sorted(unique.values(),key=lambda e:hashlib.sha256(f"0:{e['map_name']}:{e['object_id']}".encode()).digest())[:args.smoke_episodes]
        for number,entry in enumerate(selected,1):
            record=ds[entry['index']]
            prefix=f'{partition}_{entry["index"]}'
            completed=directory/f'{prefix}_complete.json'
            if completed.exists():
                info=json.loads(completed.read_text())
                if info['signature_sha256']!=signature_sha:
                    raise ValueError('Episode cache signature mismatch')
                for item in info['entries']:
                    path=directory/item['filename']
                    if not path.exists() or sha(path)!=item['sha256']:
                        raise ValueError(f'Completed cache row missing/changed: {path}')
                    entries.append(dict(item,split=partition,episode=entry['index']))
                    counters[partition].update(item['counts'])
                continue
            episode=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),
                                   tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
            policy.set_episode(episode,builder.transform(record.map_name),catalog)
            local=[]
            def collect(state,instruction):
                prediction=policy.predict(state,instruction)
                path=directory/f'{prefix}_{state.step}.pt'
                if path.exists():
                    cached=torch.load(path,map_location='cpu',weights_only=False)
                    if cached['signature_sha256']!=signature_sha:
                        raise ValueError('Partial cache signature mismatch')
                    if not torch.allclose(cached['public']['spatial']['base_logits'],policy.pack['spatial']['base_logits'].cpu(),atol=1e-5,rtol=0):
                        raise ValueError('Causal resume changed A0 dynamic belief')
                    row=cached
                else:
                    public={g:{k:v.detach().cpu().clone() for k,v in values.items()} for g,values in policy.pack.items()}
                    labels,private=label_view(record,catalog,index,public,policy.meta,state.current_observation,partition)
                    row=dict(public=public,labels=labels,private=private,meta=policy.meta,
                             target_instance=[record.map_name,record.object_id],episode=entry['index'],
                             step=state.step,instruction=instruction,signature_sha256=signature_sha)
                    temporary=path.with_suffix('.tmp');torch.save(row,temporary);temporary.replace(path)
                counts=view_counts(row)
                item=dict(filename=path.name,sha256=sha(path),step=state.step,counts=dict(counts))
                local.append(item)
                return prediction
            if partition=='calibration':
                poses,termination,seconds,_=rollout(episode,builder,controller,collect,GROUND_LEVEL[record.map_name],
                                                   ClosedLoopProtocol(),synchronize=torch.cuda.synchronize)
                trace=dict(poses=poses.tolist(),termination=termination,decision_seconds=seconds)
            else:
                trajectory=np.asarray(record.trajectory)
                legal=np.flatnonzero(np.isfinite(trajectory).all(-1) & (trajectory[:,2]-GROUND_LEVEL[record.map_name]>=10) &
                                    (trajectory[:,2]-GROUND_LEVEL[record.map_name]<=100))
                steps=sorted({int(legal[round(q*(len(legal)-1))]) for q in (.25,.75,1.)}) if len(legal) else []
                for step in steps:
                    state=builder.build_from_poses(episode,trajectory[:step+1])
                    collect(state,record.instruction)
                trace=dict(teacher_steps=steps,no_legal_observation=not bool(steps))
            save(dict(signature_sha256=signature_sha,entries=local,trace=trace),completed)
            for item in local:
                entries.append(dict(item,split=partition,episode=entry['index']))
                counters[partition].update(item['counts'])
            progress=dict(status='cache_smoke_running' if smoke else 'extracting',partition=partition,
                          episodes_completed=number,episodes_total=len(selected),
                          counters={s:dict(c) for s,c in counters.items()},seconds=time.perf_counter()-started)
            save(progress,OUTPUT/('cache_smoke_progress.json' if smoke else 'preparation_progress.json'))
            print(json.dumps(progress),flush=True)
        print(f'finished {partition}',flush=True)
    manifest=dict(complete=not smoke,signature_sha256=signature_sha,entries=entries,
                  counters={s:dict(c) for s,c in counters.items()},seconds=time.perf_counter()-started)
    save(manifest,directory/'manifest.json')
    save(dict(status='partial_cache_smoke_complete' if smoke else 'full_cache_complete',**{k:v for k,v in manifest.items() if k!='entries'}),
         ROOT/'reports/a2_execution_20261027'/('cache_smoke.json' if smoke else 'cache_audit.json'))


if __name__=='__main__':
    main()
