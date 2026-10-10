"""Seen-only observed RGB retrieval diagnostic, independent of navigation policy.

Cameras use fixed human trajectory fractions and a pose-only legal descent.
All candidate crops come from the current frame. Target ID is used exclusively
after scoring to define retrieval labels and same-category hard negatives.
No trained navigation accuracy or attribute-classification accuracy is claimed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import atomic_json,file_sha256,provenance
from graphnav.landmark_projection import project_landmark
from graphnav.observed_patch_scales import ObservedPatchScaleEncoder
from graphnav.observed_scales import extract_observed_scale
from graphnav.renderer import CityNavObservedViewRenderer


METHODS=('native_single32','native_crop_multiscale','native_patch_multiscale')


def retrieval(scores,positive):
    """Expected top1 correctness under ties; never favor a fixed positive index."""
    scores=np.asarray(scores,dtype=float)
    top=scores.max();tied=np.isclose(scores,top,rtol=0,atol=1e-7)
    return dict(r1=float(tied[positive]/max(1,tied.sum())),
        margin=float(scores[positive]-np.delete(scores,positive).max()))


def record_invalid_height(rows, stream, *, index, step, map_name, height):
    row=dict(index=int(index),step=step,map_name=map_name,view='before',
        height_m=float(height),eligible=False,reason='nonpositive_height')
    rows.append(row)
    stream.write(json.dumps(row)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('sbf-source','data-root','output','main-workflow-status'):p.add_argument('--'+key,required=True)
    p.add_argument('--episodes',type=int,default=32);p.add_argument('--microbatch',type=int,default=8)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing RGB diagnostic overwrite')
    out.mkdir(parents=True);(out/'comparisons').mkdir()
    sys.path.insert(0,a.sbf_source)
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.mapdata import GROUND_LEVEL
    from sbfnav.validation import evenly_spaced_indices
    dataset=CityNavDataset(a.data_root,'val_seen',version='processed_citynav')
    catalog=CityReferCatalog(Path(a.data_root)/'cityrefer')
    indices=evenly_spaced_indices(len(dataset),a.episodes)
    renderer=CityNavObservedViewRenderer(Path(a.data_root)/'rgbd',out/'unused',image_size=384)
    native=ObservedSigLIPMatchingEncoder(microbatch=a.microbatch)
    patch_encoder=ObservedPatchScaleEncoder(native)
    rows=[];pairs=[];visuals=[];attempted=0;start=time.perf_counter();stopped=False
    objects={};caption_buckets=defaultdict(list)
    for record in dataset:
        category=catalog.target_object(record).get('object_type','Unknown')
        caption_buckets[(record.map_name.split('_')[0],category)].append((record.object_id,record.instruction))
    def gallery(observation,name):
        if name not in objects:
            packed=[]
            for oid,obj in catalog.objects[name].items():
                poly=np.asarray(obj['contour'],np.float32)
                if poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3 or not np.isfinite(poly).all():continue
                packed.append((oid,obj,poly.min(0),poly.max(0)))
            objects[name]=packed
        footprint=np.asarray(observation['footprint_world']);lo=footprint.min(0);hi=footprint.max(0)
        result=[]
        for oid,obj,low,high in objects[name]:
            if np.any(high<lo) or np.any(low>hi):continue
            projected=project_landmark(dict(obj,id=str(oid)),footprint,observation['bgr'].shape,min_visible=.5,min_pixels=4)
            if projected is None:continue
            center=np.asarray(obj['contour'],np.float32).mean(0)
            crops={extent:extract_observed_scale(observation,center,extent_m=extent) for extent in (32.,64.,128.)}
            if crops[32.] is None:continue
            result.append(dict(id=int(oid),category=obj.get('object_type','Unknown'),center=center,
                native_object_pixels=projected.visible_pixels,crops={k:v for k,v in crops.items() if v is not None}))
        return result
    with (out/'cases.jsonl').open('w') as stream:
        for raw_index in indices:
            record=dataset[int(raw_index)];name=record.map_name;ground=GROUND_LEVEL[name]
            steps=sorted({0,int(.4*(len(record.trajectory)-1)),int(.8*(len(record.trajectory)-1))})
            for step in steps:
                status=json.loads(Path(a.main_workflow_status).read_text())
                if status.get('stage')!='prepare_frozen_features' or status.get('status')!='running':
                    stopped=True;break
                pose=record.trajectory[step].copy();height=pose[2]-ground
                if height<=0:
                    attempted+=1
                    record_invalid_height(rows,stream,index=raw_index,step=step,map_name=name,height=height)
                    continue
                poses=[('before',pose)]
                if 10.<height<=100.:
                    descended=pose.copy();descended[2]-=min(10.,height-10.);poses.append(('descend10',descended))
                scored={}
                for view,where in poses:
                    attempted+=1
                    observation=renderer.current_rgbd(name,where,ground)
                    candidates=gallery(observation,name)
                    base=dict(index=int(raw_index),map_name=name,instruction=record.instruction,step=step,
                        view=view,height_m=float(where[2]-ground),eligible=False,gallery_all_categories=len(candidates))
                    if not candidates:
                        row=dict(base,reason='no_supported32m_candidate_ROI');rows.append(row);stream.write(json.dumps(row)+'\n');continue
                    crop_images=[];locations=[]
                    for ci,candidate in enumerate(candidates):
                        for scale,crop in candidate['crops'].items():crop_images.append(crop.bgr);locations.append((ci,scale))
                    vectors=native.images(crop_images)
                    images={}
                    for (ci,scale),vector in zip(locations,vectors):images.setdefault(ci,{})[scale]=vector
                    patch,valid,_=patch_encoder.encode([observation],[np.stack([c['center'] for c in candidates])])[0]
                    query=native.texts([record.instruction])[0]
                    scores={method:{} for method in METHODS}
                    for ci,candidate in enumerate(candidates):
                        if not valid[ci,0]:continue # Same observed32m support for all methods.
                        oid=candidate['id']
                        scores['native_single32'][oid]=float(query@images[ci][32.])
                        scores['native_crop_multiscale'][oid]=float(torch.stack([query@v for v in images[ci].values()]).mean())
                        scores['native_patch_multiscale'][oid]=float((F.normalize(patch[ci,valid[ci]].to(query.device).float(),dim=-1)@query).mean())
                    # Destination identity/category enters only evaluator bookkeeping.
                    category=catalog.target_object(record).get('object_type','Unknown')
                    ids=sorted(c['id'] for c in candidates if c['category']==category and c['id'] in scores[METHODS[0]])
                    base.update(category=category,same_category_candidates=len(ids))
                    if record.object_id not in ids or len(ids)<2:
                        row=dict(base,reason='target_unobserved_or_fewer_than_two_same_category_candidates')
                    else:
                        positive=ids.index(record.object_id)
                        measures={method:retrieval([scores[method][oid] for oid in ids],positive) for method in METHODS}
                        row=dict(base,eligible=True,reason=None,metrics=measures,
                            target_native_pixels=next(c['native_object_pixels'] for c in candidates if c['id']==record.object_id),
                            description_groups=[group for group,pattern in [('color_words',r'\b(red|brown|grey|gray|white|black|blue|green)\b'),
                                ('shape',r'\b(square|rectangular|round|circular|triangular|shaped)\b'),('vehicle',r'\b(car|cars|vehicle|vehicles|truck)\b')]
                                if re.search(pattern,record.instruction,re.I)])
                        # Same-city/type wrong language control, chosen independently of RGB scores.
                        wrong=next((text for oid,text in caption_buckets[(name.split('_')[0],category)]
                                    if oid!=record.object_id and text!=record.instruction),None)
                        if wrong is not None:
                            wrong_query=native.texts([wrong])[0]
                            wrong_scores=[float(wrong_query@images[next(i for i,c in enumerate(candidates) if c['id']==oid)][32.]) for oid in ids]
                            row['wrong_language_single32']=retrieval(wrong_scores,positive)
                        if len(visuals)<24:
                            chosen=max(ids,key=lambda oid:scores['native_single32'][oid])
                            true_crop=next(c['crops'][32.].bgr for c in candidates if c['id']==record.object_id)
                            pred_crop=next(c['crops'][32.].bgr for c in candidates if c['id']==chosen)
                            montage=np.concatenate((observation['bgr'],true_crop,pred_crop),axis=1)
                            filename=f'{len(visuals):03d}_ep{int(raw_index)}_step{step}_{view}.png'
                            cv2.imwrite(str(out/'comparisons'/filename),montage)
                            visuals.append(dict(file=filename,panels=['actual384_frame','label_ROI32_digital_crop','native_top1_ROI32_digital_crop'],
                                instruction=record.instruction,index=int(raw_index),step=step,view=view))
                    rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    scored[view]=(scores,ids)
                if set(scored)=={'before','descend10'}:
                    ids=sorted(set(scored['before'][1])&set(scored['descend10'][1]))
                    if record.object_id in ids and len(ids)>=2:
                        positive=ids.index(record.object_id)
                        pairs.append(dict(index=int(raw_index),step=step,candidates=len(ids),
                            metrics={view:{method:retrieval([scored[view][0][method][oid] for oid in ids],positive)
                                          for method in METHODS} for view in scored}))
                atomic_json(dict(status='running',attempted_observations=attempted,eligible=sum(r['eligible'] for r in rows),
                    seconds=time.perf_counter()-start),out/'status.json')
            if stopped:break
    eligible=[r for r in rows if r['eligible']]
    def summarize(items):
        return dict(cases=len(items),**{method:{metric:float(np.mean([r['metrics'][method][metric] for r in items])) if items else None
            for metric in ('r1','margin')} for method in METHODS})
    summary=dict(status='interrupted_before_training' if stopped else 'complete',args=vars(a),provenance=provenance(),
        diagnostic_script_sha256=file_sha256(__file__),
        source_sha256=file_sha256(dataset.annotation_path),backbone_revision=getattr(native.model.config,'_commit_hash',None),
        selected_indices=[int(i) for i in indices],attempted_observations=attempted,eligible_observations=len(eligible),
        eligible_fraction=len(eligible)/max(attempted,1),all_eligible=summarize(eligible),
        by_category={c:summarize([r for r in eligible if r['category']==c]) for c in sorted({r['category'] for r in eligible})},
        by_description={g:summarize([r for r in eligible if g in r['description_groups']]) for g in ('color_words','shape','vehicle')},
        by_view={v:summarize([r for r in eligible if r['view']==v]) for v in ('before','descend10')},
        paired_descent={method:{view:float(np.mean([r['metrics'][view][method]['r1'] for r in pairs])) if pairs else None
                                for view in ('before','descend10')} for method in METHODS},
        query_macro_r1={method:float(np.mean([np.mean([r['metrics'][method]['r1'] for r in eligible if (r['map_name'],r['instruction'])==q])
                         for q in sorted({(r['map_name'],r['instruction']) for r in eligible})])) if eligible else None for method in METHODS},
        wrong_language_single32_r1=float(np.mean([r['wrong_language_single32']['r1'] for r in eligible if 'wrong_language_single32' in r]))
            if any('wrong_language_single32' in r for r in eligible) else None,
        paired_same_gallery_cases=len(pairs),seconds=time.perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,encoded_images=native.encoded_images,
        caveats=['Offline conditional target-instance retrieval within same-category actual observed candidates; no SR/SPL claim.',
            'Projected polygons are imperfect geometric proposals, not verified segmentation.',
            'Cameras follow fixed fractions; descent uses public pose only. No future human frame or GT camera selection.',
            'ROI resizing is digital resampling; patch method includes current-frame contextual attention.',
            'Description groups are language mentions, not human-verified color/shape classification labels.',
            'Existing P2 head was trained for cross-frame instance/type alignment; no verified full-language matching head used.',
            'Concurrent with training-feature preparation; measured speed is shared-resource diagnostic timing.'])
    atomic_json(summary,out/'summary.json');atomic_json(pairs,out/'paired_descent.json');atomic_json(visuals,out/'visualizations.json')
    atomic_json(dict(status=summary['status'],attempted_observations=attempted,eligible=len(eligible)),out/'status.json')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
