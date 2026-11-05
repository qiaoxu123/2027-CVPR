"""Full Polygon swaps, perturbations and causal reference matching on frozen RGB."""
import argparse
from collections import OrderedDict
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
from shapely import affinity
from scripts.polygon_rgb_matching import CONFIG,candidate_inputs,comparable,image_evidence,explicit_score,score_contrast


def contours(poly):
    parts=[poly] if poly.geom_type=='Polygon' else poly.geoms
    return [list(map(list,p.exterior.coords)) for p in parts if p.geom_type=='Polygon']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('nearfield','rollout','data-root','sbf-source','checkpoint','stage1','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();sys.path.insert(0,str(a.sbf_source))
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.renderer import CityNavObservedViewRenderer
    from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
    from graphnav.polygon_guided_visual import PolygonGuidedVisualBelief
    from graphnav.pgvb_observation import encode_landmark_observation
    from graphnav.landmark_projection import frame_to_world,world_to_frame
    from graphnav.local_landmark_anchor import _mask
    from graphnav.landmark_alias import canonical_name
    from graphnav.landmark_region_supervision import public_type
    from graphnav.observed_patch_scales import masked_native_pool
    from graphnav.checkpointing import file_sha256
    torch.set_num_threads(4);torch.manual_seed(42)
    if a.output.exists() and not a.resume:raise ValueError('New output or --resume required')
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'queries').mkdir(exist_ok=True)
    manifest=dict(config=CONFIG,checkpoint_sha256=file_sha256(a.checkpoint),stage1_sha256=file_sha256(a.stage1),
        dataset_sha256=file_sha256(a.nearfield/'dataset.jsonl'),rollout_sha256=file_sha256(a.rollout/'predictions.jsonl'),
        source_sha256={f:file_sha256(f) for f in [__file__,'scripts/polygon_rgb_matching.py','graphnav/local_landmark_anchor.py','graphnav/polygon_guided_visual.py']},
        label_policy='Public-map instance IDs and synthetic perturbations are weak proxies, no independent masks',
        query_name='fixed A; each candidate swaps full Polygon, tokens, ROI labels and pooled visual region',
        negative_caveat='another real Polygon B may genuinely align with RGB; identity retrieval is distinct from alignment',
        reference='strictly earlier visited frame, minimum 1m XY change or 5deg yaw; no future RGB')
    mp=a.output/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    stage=torch.load(a.stage1,map_location='cpu',weights_only=True);bundle=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    native=ObservedSigLIPMatchingEncoder(stage['protocol']['backbone'],device='cuda',microbatch=1)
    assert native.model.config._commit_hash==stage['protocol']['backbone_revision']
    head=PolygonGuidedVisualBelief().cuda().eval().requires_grad_(False)
    head.load_state_dict({k[5:]:v for k,v in bundle['model'].items() if k.startswith('pgvb.')},strict=True)
    renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'render',384)
    frames=list(map(json.loads,(a.nearfield/'dataset.jsonl').open()));frames.sort(key=lambda r:(r['index'],r['step']))
    catalog=json.loads((a.data_root/'cityrefer/objects.json').read_text())
    rollouts={r['index']:r for r in map(json.loads,(a.rollout/'predictions.jsonl').open()) if r['method']=='StableTrackedPGVB'}
    cache=OrderedDict();calls=0;hits=0;start=time.perf_counter();torch.cuda.reset_peak_memory_stats();output=[]
    def encode(index,step,meta,kind):
        nonlocal calls,hits
        key=(index,step)
        if key in cache:hits+=1;cache.move_to_end(key);return cache[key]
        r=rollouts[index];root=a.rollout/'decisions'/f'{index:02d}_StableTrackedPGVB'/f'step_{step:02d}'
        snap=np.load(root.with_suffix('.npz'));bgr=cv2.imread(str(root.with_suffix('.png')))
        obs=renderer.current_rgbd(r['map_name'],r['poses'][step],GROUND_LEVEL[r['map_name']]);assert np.array_equal(obs['bgr'],bgr)
        saved={};hook=native.model.vision_model.register_forward_hook(lambda m,i,o:saved.update(patches=o.last_hidden_state[0].detach()))
        inputs,reason=encode_landmark_observation(obs,dict(name=r['landmark_name'],typ=kind,contours=meta['anchor_contours']),dict(world_bounds=snap['bounds'].tolist()),native,native.device)
        hook.remove()
        calls+=int(inputs is not None)
        result=dict(obs=obs,inputs=inputs,reason=reason,patches=saved.get('patches'),rgb_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),bounds=snap['bounds'])
        cache[key]=result
        while len(cache)>12:cache.popitem(last=False)
        return result
    for frame in frames:
        key=frame['key'];path=a.output/'queries'/f'{key}.json'
        if path.exists() and a.resume:output.append(json.loads(path.read_text()));continue
        index=frame['index'];step=frame['step'];meta=json.loads((a.nearfield/'maps'/f'{index:02d}.json').read_text());kind=frame['landmark_type']
        row=dict(key=key,index=index,step=step,map_name=frame['map_name'],query_name=frame['landmark_name'],kind=kind,height_agl_m=frame['height_agl_m'],pose=frame['pose'],
                 candidates=[],reference=None,eligibility='pending',independent_mask_GT=False)
        if kind not in ('Building','Road'):
            row['eligibility']='unsupported_PGVB_type';path.write_text(json.dumps(row)+'\n');output.append(row);continue
        with torch.inference_mode():
            view=encode(index,step,meta,kind);obs=view['obs'];base=candidate_inputs(meta['anchor_contours'],obs)
            if view['inputs'] is None or not base['eligible']:
                row['eligibility']='insufficient_anchor_RGB_or_positive_negative_support';row['reasons']=base['reasons'];path.write_text(json.dumps(row)+'\n');output.append(row);continue
            row.update(eligibility='eligible',rgb_sha256=view['rgb_sha256'],footprint_world=np.asarray(obs['footprint_world']).tolist())
            anchor=unary_union([Polygon(c) for c in meta['anchor_contours']]);groups={}
            for oid,obj in catalog[frame['map_name']].items():
                if public_type([obj['object_type']])!=kind or str(oid) in meta['anchor_ids']:continue
                poly=Polygon(obj['contour'])
                if not poly.is_valid or poly.area<=0:continue
                group=canonical_name(obj.get('name','')) or 'unnamed:'+str(oid)
                if group==canonical_name(frame['landmark_name']):continue
                groups.setdefault(group,[]).append((str(oid),obj,poly))
            negatives=[];fp=Polygon(obs['footprint_world'])
            for group,items in groups.items():
                poly=unary_union([item[2] for item in items])
                if not poly.intersects(fp) or anchor.intersection(poly).area/max(anchor.union(poly).area,1e-9)>.1:continue
                cs=[item[1]['contour'] for item in items]
                ci=candidate_inputs(cs,obs)
                if comparable(base,ci):
                    difficulty=abs(np.log(ci['anchor']['local_geometry'].area/base['anchor']['local_geometry'].area))+abs(ci['anchor']['whole_fraction']-base['anchor']['whole_fraction'])
                    negatives.append((difficulty,group,poly,ci,[x[0] for x in items],cs))
            candidates=[dict(id='A',family='correct',polygon=anchor,cs=meta['anchor_contours'],ci=base,public_ids=meta['anchor_ids'],name=frame['landmark_name'])]
            for _,group,poly,ci,ids,cs in sorted(negatives,key=lambda x:(x[0],x[1]))[:3]:
                candidates.append(dict(id='B:'+group,family='real_other_instance',polygon=poly,cs=cs,ci=ci,public_ids=ids,name=group))
            center=np.array([192.,192.]);world=frame_to_world(np.stack([center,center+[1,0],center+[0,1]]),obs['footprint_world'],(384,384,3));dx=world[1]-world[0];dy=world[2]-world[0]
            proposed=[]
            origin=anchor.centroid.coords[0]
            def propose(label,family,fn):
                proposed.append((label,family,fn(anchor),[contours(fn(Polygon(cc)))[0] for cc in meta['anchor_contours']]))
            for pixels in CONFIG['pixel_shifts']:
                for axis,v in [('x',dx),('y',dy)]:
                    for sign in (-1,1):
                        shift=v*pixels*sign;propose(f'shift_{axis}_{sign*pixels}px','position_shift',lambda p:affinity.translate(p,xoff=shift[0],yoff=shift[1]))
            for deg in CONFIG['rotation_degrees']:propose(f'rotate_{deg}','shape_rotation',lambda p:affinity.rotate(p,deg,origin=origin))
            for scale in CONFIG['scale_factors']:propose(f'scale_{scale}','shape_scale',lambda p:affinity.scale(p,xfact=scale,yfact=1/scale,origin=origin))
            row['excluded_perturbations']=[]
            for label,family,poly,cs in proposed:
                ci=candidate_inputs(cs,obs)
                if comparable(base,ci):candidates.append(dict(id=label,family=family,polygon=poly,cs=cs,ci=ci,public_ids=[],name=frame['landmark_name']))
                else:row['excluded_perturbations'].append(label)
            # Causal reference A: public registration, earlier actual image only.
            reference=None
            for old in range(step-1,-1,-1):
                oldpose=np.array(rollouts[index]['poses'][old]);pose=np.array(frame['pose']);angle=abs((pose[3]-oldpose[3]+np.pi)%(2*np.pi)-np.pi)
                if np.linalg.norm(pose[:2]-oldpose[:2])<1 and angle<np.deg2rad(5):continue
                rview=encode(index,old,meta,kind)
                if rview['inputs'] is None:continue
                reference=rview;row['reference']=dict(step=old,rgb_sha256=rview['rgb_sha256'],pose=oldpose.tolist(),source='earlier_actual_same_map_ID_proxy');break
            features=view['inputs'];n=len(candidates)
            arguments={k:v.expand(n,*v.shape[1:]) for k,v in features.items() if k not in ('local_uv','polygon_tokens')}
            arguments['polygon_tokens']=torch.tensor(np.stack([c['ci']['tokens'] for c in candidates]),device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scores=head(**arguments).float().cpu().numpy()
                zero_scores=head(**{**arguments,'polygon_tokens':torch.zeros_like(arguments['polygon_tokens'])}).float().cpu().numpy()
            evidence=image_evidence(obs['bgr'],obs['depth_valid']&(obs['bgr'].max(-1)>0))
            shifted_evidence=image_evidence(np.roll(obs['bgr'],96,axis=0),obs['depth_valid']&(obs['bgr'].max(-1)>0))
            def pool(roi,robs,patches):
                mask=_mask(roi.intersection(Polygon(robs['footprint_world'])),robs['footprint_world'])
                support=robs['depth_valid']&(robs['bgr'].max(-1)>0)
                pm=((mask&support).reshape(24,16,24,16).mean((1,3)).ravel()>=.5)
                if pm.sum()<2:return None
                with torch.autocast('cuda',dtype=torch.bfloat16):return F.normalize(masked_native_pool(native.model.vision_model.head,patches,torch.tensor(pm[None],device='cuda')).float(),dim=-1)[0]
            refvector=pool(anchor,reference['obs'],reference['patches']) if reference else None
            vectors=[]
            for j,candidate in enumerate(candidates):
                ci=candidate['ci'];labels=ci['anchor']['labels'];valid=features['valid'][0].cpu().numpy()&ci['anchor']['valid_patches']
                explicit=explicit_score(ci['anchor']['geometry_mask'],evidence);shifted=explicit_score(ci['anchor']['geometry_mask'],shifted_evidence)
                vector=pool(candidate['polygon'],obs,view['patches']);appearance=float(vector@refvector) if vector is not None and refvector is not None else None
                pgvb=score_contrast(scores[j],labels,valid);zero=score_contrast(zero_scores[j],labels,valid)
                row['candidates'].append(dict(id=candidate['id'],family=candidate['family'],public_ids=candidate['public_ids'],candidate_name=candidate['name'],query_name=frame['landmark_name'],
                    contours=candidate['cs'],polygon_tokens=ci['tokens'].tolist(),
                    roi_mask_sha256=hashlib.sha256(ci['anchor']['geometry_mask'].tobytes()).hexdigest(),
                    positive_candidate_uv=features['local_uv'][0][torch.as_tensor((labels==1)&valid,device='cuda')].cpu().numpy().tolist(),
                    local_area_m2=ci['anchor']['local_geometry'].area,visible_fraction=ci['anchor']['whole_fraction'],positive_patches=ci['anchor']['positive_patches'],negative_patches=ci['anchor']['negative_patches'],
                    geometry=ci['geometry_score'],pgvb=pgvb,pgvb_zero_shape_tokens=zero,
                    explicit=explicit['score'] if explicit else None,explicit_components=explicit,
                    explicit_shifted_RGB=shifted['score'] if shifted else None,appearance=appearance,
                    explicit_reference=(.8*appearance+.2*explicit['score']) if appearance is not None and explicit is not None else None,
                    shape_token_score_MAE=float(abs(scores[j]-zero_scores[j]).mean())))
                vectors.append(vector.cpu().numpy() if vector is not None else np.zeros(1024))
            row['reference_cosine_correct']=row['candidates'][0]['appearance']
            # Correct overlap denominator is A's observed portion, never the last candidate.
            if reference:row['reference_visible_overlap_fraction']=float(anchor.intersection(Polygon(obs['footprint_world'])).intersection(Polygon(reference['obs']['footprint_world'])).area/max(base['anchor']['local_geometry'].area,1e-9))
            np.savez_compressed(a.output/'queries'/f'{key}.npz',vectors=np.array(vectors),reference_vector=refvector.cpu().numpy() if refvector is not None else np.zeros(1024))
            path.write_text(json.dumps(row,ensure_ascii=False)+'\n');output.append(row)
        print(json.dumps(dict(key=key,queries=len(output),candidates=len(row['candidates']),seconds=time.perf_counter()-start)),flush=True)
    (a.output/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    (a.output/'runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,queries=len(output),native_rgb_forwards=calls,cache_hits=hits,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,checkpoint_sha256_after=file_sha256(a.checkpoint)),indent=2)+'\n')
    assert file_sha256(a.checkpoint)==manifest['checkpoint_sha256']

if __name__=='__main__':main()
