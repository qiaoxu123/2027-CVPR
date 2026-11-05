"""Offline nearfield scene audit on visited frozen rollouts; no policy changes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scripts.nearfield_geometry import legal_polygon, neighbors, relation, road_enclosures

CLASSES = ('Building', 'Road', 'Parking', 'Vegetation', 'Car', 'Ground', 'Water', 'Other')
PROMPTS = ('an aerial photograph of a building roof', 'an aerial photograph of a road',
           'an aerial photograph of a parking lot', 'an aerial photograph of trees and vegetation',
           'an aerial photograph of a car', 'an aerial photograph of open ground',
           'an aerial photograph of water', 'an aerial photograph of other structures')
GROUP = dict(Road='Road',Other='Other',Building='Building',TrafficRoad='Road',Footpath='Road',Bridge='Road',Parking='Parking',
             HighVegetation='Vegetation',Car='Car',Ground='Ground',Water='Water')


def auc(scores, labels, valid):
    pos=scores[(labels==1)&valid];neg=scores[(labels==0)&valid]
    if len(pos)<2 or len(neg)<2:return None
    diff=pos[:,None]-neg[None,:]
    return float(((diff>0)+.5*(diff==0)).mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','selection','data-root','sbf-source','stage1','checkpoint','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.renderer import CityNavObservedViewRenderer
    from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
    from graphnav.pgvb_observation import encode_landmark_observation
    from graphnav.polygon_guided_visual import PolygonGuidedVisualBelief
    from graphnav.local_landmark_anchor import named_public_polygons, local_landmark_anchor, _mask
    from graphnav.landmark_region_supervision import public_type
    from graphnav.landmark_projection import frame_to_world
    from graphnav.observed_patch_scales import masked_native_pool
    from graphnav.checkpointing import file_sha256
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    if a.output.exists() and not a.resume:raise ValueError('Use a new output or --resume')
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'frames').mkdir(exist_ok=True);(a.output/'maps').mkdir(exist_ok=True)
    rows=[r for r in map(json.loads,(a.run/'predictions.jsonl').open()) if r['method']=='StableTrackedPGVB'];assert len(rows)==64
    tasks={t['index']:t for t in json.loads((a.selection/'tasks_public.json').read_text())}
    catalog=json.loads((a.data_root/'cityrefer/objects.json').read_text())
    sources=dict(checkpoint_sha256=file_sha256(a.checkpoint),stage1_sha256=file_sha256(a.stage1),
        rollout_sha256=file_sha256(a.run/'predictions.jsonl'),tasks_sha256=file_sha256(a.selection/'tasks_public.json'),
        catalog_sha256=file_sha256(a.data_root/'cityrefer/objects.json'),source_commit='97c705581104fcdfd58ac806bc3b442fa1cce6ea',
        class_prompts=list(PROMPTS),near_radius_m=100,near_pose_distance_m=20,
        weak_roi_min_pixels=64,weak_roi_min_native_patches=2,
        visibility='geometric_projection_and_source_support_only; actual_visibility_unknown',
        annotation_policy='No independent human masks or identity labels; every accuracy is proxy or unavailable',
        inference_inputs='visited RGB, pose, footprint, public map only; no original CityNav target GT',
        source_sha256={name:file_sha256(name) for name in [__file__,'scripts/nearfield_geometry.py','graphnav/pgvb_observation.py','graphnav/polygon_guided_visual.py','graphnav/observed_patch_scales.py','graphnav/local_landmark_anchor.py','graphnav/renderer.py']})
    fp=a.output/'manifest.json'
    if fp.exists():assert json.loads(fp.read_text())==sources
    fp.write_text(json.dumps(sources,indent=2)+'\n')
    stage=torch.load(a.stage1,map_location='cpu',weights_only=True)
    native=ObservedSigLIPMatchingEncoder(stage['protocol']['backbone'],device='cuda',microbatch=1)
    assert native.model.config._commit_hash==stage['protocol']['backbone_revision']
    head=PolygonGuidedVisualBelief().cuda().eval().requires_grad_(False)
    bundle=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    head.load_state_dict({k[len('pgvb.'):]:v for k,v in bundle['model'].items() if k.startswith('pgvb.')},strict=True)
    texts=native.texts(list(PROMPTS));renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'render_tmp',image_size=384)
    selected=[];all_rows=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    # A single zero-image transformer pass, with the ORIGINAL per-ROI masks retained.
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        zero_batch=native.processor(images=[np.zeros((384,384,3),np.uint8)],return_tensors='pt').to('cuda')
        zero_patches=native.model.vision_model(**zero_batch).last_hidden_state[0]
    for r in sorted(rows,key=lambda r:r['index']):
        index=r['index'];task=tasks[index];objects=catalog[r['map_name']]
        binding=named_public_polygons(objects,r['landmark_name']);kind=public_type([objects[i]['object_type'] for i in binding['ids']])
        anchor=unary_union([Polygon(c) for c in binding['contours']])
        neighborhood=neighbors(objects,anchor,set(binding['ids']))
        roads=[(oid,poly) for oid,obj,poly in neighborhood if GROUP.get(obj['object_type'])=='Road']
        roads += [(oid,legal_polygon(objects[oid]['contour'])) for oid in binding['ids'] if GROUP.get(objects[oid]['object_type'])=='Road']
        enclosures=road_enclosures(roads)
        map_rows=[dict(public_id=oid,name=obj.get('name',''),type=obj['object_type'],contour=obj['contour'],
                       relation_from_anchor=relation(anchor,poly)) for oid,obj,poly in neighborhood]
        nearest=[];chains=[]
        for oid,obj,poly in neighborhood:
            if obj['object_type']=='Building' and roads:
                dist=min(poly.distance(q) for _,q in roads)
                nearest.append(dict(building_id=oid,nearest_road_ids=[rid for rid,q in roads if abs(poly.distance(q)-dist)<1e-6],distance_m=dist,evidence='map_only_nearest_road_not_language_beside_label'))
            if obj['object_type']=='Car':
                for pid,parking,pp in neighborhood:
                    if parking['object_type']=='Parking' and pp.covers(poly):
                        chains.append(dict(anchor_name=r['landmark_name'],parking_id=pid,car_id=oid,evidence='public_polygons_only',visual_chain_verified=False))
        map_record=dict(index=index,map_name=r['map_name'],landmark_name=r['landmark_name'],landmark_type=kind,
                        anchor_ids=binding['ids'],anchor_contours=binding['contours'],neighbors=map_rows,
                        building_nearest_road=nearest,car_inside_parking_chains=chains,road_enclosures=enclosures,
                        no_enclosure_means='unknown_if_no_exact_hole; no inferred missing road completion')
        (a.output/'maps'/f'{index:02d}.json').write_text(json.dumps(map_record,ensure_ascii=False)+'\n')
        near=[i for i,d in enumerate(r['step_diagnostics']) if d['near_landmark_geometry']['within_20m']]
        tracked_near=[i for i in near if r['step_diagnostics'][i]['landmark_tracking_state']=='tracking']
        chosen=sorted(set(([near[0],near[-1]] if near else [len(r['step_diagnostics'])-1])+([tracked_near[0]] if tracked_near else [])))
        for step in chosen:
            key=f'{index:02d}_{step:02d}';selected.append(dict(key=key,index=index,step=step,near=step in near))
            output=a.output/'frames'/f'{key}.json'
            if output.exists() and output.with_suffix('.png').exists() and output.with_suffix('.npz').exists() and a.resume:all_rows.append(json.loads(output.read_text()));continue
            source=a.run/'decisions'/f'{index:02d}_StableTrackedPGVB'/f'step_{step:02d}'
            snapshot=np.load(source.with_suffix('.npz'));d=r['step_diagnostics'][step]
            bgr=cv2.imread(str(source.with_suffix('.png')));rgbsha=hashlib.sha256(bgr.tobytes()).hexdigest();assert rgbsha==d['rgb_sha256']
            observation=renderer.current_rgbd(r['map_name'],r['poses'][step],GROUND_LEVEL[r['map_name']])
            assert np.array_equal(bgr,observation['bgr']);assert np.allclose(snapshot['footprint_world'],observation['footprint_world'])
            record=dict(world_bounds=task['world_bounds']);saved={}
            hook=native.model.vision_model.register_forward_hook(lambda module,inputs,result:saved.update(patches=result.last_hidden_state[0].detach()))
            with torch.inference_mode():
                inputs,reason,metadata=encode_landmark_observation(observation,dict(name=r['landmark_name'],typ=kind,contours=binding['contours']) if kind in ('Building','Road') else None,record,native,native.device,return_metadata=True)
                if inputs is None:
                    batch=native.processor(images=[cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)],return_tensors='pt').to('cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):native.model.vision_model(**batch)
                patches=saved['patches'];hook.remove()
                anchor_audit=dict(reason=reason,independent_identity_verified=False,actual_recognition_accuracy=None)
                if inputs is not None:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        real=head(**{k:v for k,v in inputs.items() if k not in ('local_uv',)}).float()
                        shuffled=head(**{**{k:v for k,v in inputs.items() if k!='local_uv'},'image_features':inputs['image_features'].roll(72,1)}).float()
                        zero=head(**{**{k:v for k,v in inputs.items() if k!='local_uv'},'image_features':torch.zeros_like(inputs['image_features'])}).float()
                        alternatives=sorted({o.get('name','') for o in objects.values() if o.get('name') and o.get('name')!=r['landmark_name'] and public_type([o['object_type']])==kind})
                        wrong_name=alternatives[0] if alternatives else None
                        wrong=head(**{**{k:v for k,v in inputs.items() if k!='local_uv'},'name_text':native.texts([wrong_name])}).float() if wrong_name else None
                    labels=metadata['anchor_labels'][0].cpu().numpy();valid=inputs['valid'][0].cpu().numpy();scores=real[0].cpu().numpy()
                    logged=snapshot['scores'];mae=float(abs(scores-logged).mean())
                    anchor_audit.update(real_auc_proxy=auc(scores,labels,valid),shuffled_auc_proxy=auc(shuffled[0].cpu().numpy(),labels,valid),zero_features_auc_proxy=auc(zero[0].cpu().numpy(),labels,valid),
                        wrong_same_type_name=wrong_name,wrong_name_auc_proxy=auc(wrong[0].cpu().numpy(),labels,valid) if wrong is not None else None,
                        logged_score_mae=mae,positive_patches=int((labels==1).sum()),negative_patches=int((labels==0).sum()),
                        top_patch_label=int(labels[np.where(valid,scores,-np.inf).argmax()]),
                        actual_scores=scores.tolist(),valid=valid.tolist(),labels=labels.tolist(),local_uv=inputs['local_uv'][0].cpu().numpy().tolist())
                # Map-assisted ROI proposals; semantic predictions do NOT receive type/name labels.
                anchor_groups={GROUP.get(objects[i]['object_type'],'Other') for i in binding['ids']}
                anchor_label=next(iter(anchor_groups)) if len(anchor_groups)==1 else 'Other'
                items=[('__anchor__',dict(name=r['landmark_name'],object_type=anchor_label),anchor)]+neighborhood
                support=observation['depth_valid']&(bgr.max(-1)>0)
                patch_support=support.reshape(24,16,24,16).mean((1,3)).ravel()>=.9
                rois=[];masks=[]
                view=Polygon(observation['footprint_world'])
                for oid,obj,poly in items:
                    local=poly.intersection(view);mask=_mask(local,observation['footprint_world']) if not local.is_empty else np.zeros((384,384),bool)
                    supported=mask&support;pm=(supported.reshape(24,16,24,16).mean((1,3)).ravel()>=.5)&patch_support
                    entry=dict(public_id=oid,name=obj.get('name',''),map_type=obj['object_type'],expected_class=GROUP.get(obj['object_type'],'Other'),
                        polygon_pixels=int(mask.sum()),supported_pixels=int(supported.sum()),native_patch_count=int(pm.sum()),
                        whole_polygon_coverage=float(local.area/poly.area),rgb_support_fraction=float(supported.sum()/max(mask.sum(),1)),
                        anchor_distance_m=float(anchor.distance(poly)),geometric_projection_only=True,visual_identity_verified=False)
                    if supported.sum()>=64 and pm.sum()>=2:
                        entry['status']='pooled_RGB_available';masks.append(pm);entry['pool_index']=len(masks)-1
                    else:entry['status']='out_of_view' if mask.sum()==0 else 'insufficient_RGB_patch_support'
                    rois.append(entry)
                if masks:
                    mask_tensor=torch.as_tensor(np.array(masks),device='cuda')
                    vectors=[];fake_vectors=[];zero_vectors=[]
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        for offset in range(0,len(masks),64):
                            mm=mask_tensor[offset:offset+64]
                            for target,pp in ((vectors,patches),(fake_vectors,patches.roll(288,0)),(zero_vectors,zero_patches)):
                                target.append(F.normalize(masked_native_pool(native.model.vision_model.head,pp,mm).float(),dim=-1))
                    similarities=torch.cat(vectors)@texts.T;fake_sim=torch.cat(fake_vectors)@texts.T;zero_sim=torch.cat(zero_vectors)@texts.T
                    for roi in rois:
                        if 'pool_index' not in roi:continue
                        q=roi.pop('pool_index');score=similarities[q].cpu().numpy();order=np.argsort(-score)
                        roi.update(predicted_class=CLASSES[order[0]],top3_classes=[CLASSES[j] for j in order[:3]],similarities=score.tolist(),
                            cosine_margin=float(score[order[0]]-score[order[1]]),shuffled_predicted_class=CLASSES[int(fake_sim[q].argmax())],
                            zero_rgb_predicted_class=CLASSES[int(zero_sim[q].argmax())],class_match_is_map_proxy_only=True)
                actual_anchor=rois[0]
                registered=actual_anchor['supported_pixels']>=256
                tracking=d['landmark_tracking_state']=='tracking'
                semantic_agrees=actual_anchor.get('predicted_class')==actual_anchor['expected_class']
                status=('RGB_supported_proxy' if tracking and semantic_agrees else
                        'conflicting_RGB_proxy' if tracking and not semantic_agrees and 'predicted_class' in actual_anchor else
                        'geometry_only' if registered else 'insufficient_visual_evidence')
                result=dict(key=key,index=index,step=step,map_name=r['map_name'],landmark_name=r['landmark_name'],landmark_type=kind,
                    instruction=r['instruction'],pose=r['poses'][step],height_agl_m=float(r['poses'][step][2]-GROUND_LEVEL[r['map_name']]),
                    footprint_world=observation['footprint_world'].tolist(),rgb_sha256=rgbsha,source_rgb=str(source.with_suffix('.png')),
                    rgb_dimensions=list(bgr.shape),near=step in near,tracking_state=d['landmark_tracking_state'],
                    tracking_audit=d['landmark_acquisition'],landmark_status_proxy=status,anchor_audit=anchor_audit,objects=rois,
                    actual_human_verified_identity_status='unknown',actual_instance_accuracy=None,
                    camera_model='orthophoto crop + vertical height proxy; no 3D occlusion visibility oracle')
                output.write_text(json.dumps(result,ensure_ascii=False)+'\n');cv2.imwrite(str(a.output/'frames'/f'{key}.png'),bgr)
                np.savez_compressed(a.output/'frames'/f'{key}.npz',footprint_world=snapshot['footprint_world'],bounds=snapshot['bounds'],pose=snapshot['pose'],
                    pgvb_probability=snapshot['pgvb_probability'] if 'pgvb_probability' in snapshot else np.zeros(784),
                    depth_valid=observation['depth_valid'])
                all_rows.append(result)
            print(json.dumps(dict(key=key,frames=len(all_rows),roi=len(rois),status=status,seconds=time.perf_counter()-start)),flush=True)
    (a.output/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    (a.output/'dataset.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows))
    (a.output/'runtime.json').write_text(json.dumps(dict(frames=len(all_rows),seconds=time.perf_counter()-start,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        checkpoint_sha256_after=file_sha256(a.checkpoint),native_revision=native.model.config._commit_hash),indent=2)+'\n')
    assert file_sha256(a.checkpoint)==sources['checkpoint_sha256']

if __name__=='__main__':main()
