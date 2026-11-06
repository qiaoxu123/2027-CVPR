"""Full Val-Seen initial-Goal accuracy at actual starts; no privileged landmark views."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.unified_pgvb_v3_runtime import V3CausalBelief
from graphnav.unified_pgvb_v3 import offline_topk
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.v3_diagnostics import visual_correction_capacity

def failed_row(index,record,error):
    """Count a failed episode without reusing any previous forward or GT."""
    return dict(correction_capacity=None,index=index,episode_id=list(record.episode_id),
                map_name=record.map_name,error=repr(error),
                **{branch:dict(NE=None,hits={str(k):False for k in (1,5,20)})
                   for branch in ('geometry','fused')})

def main():
    p=argparse.ArgumentParser();p.add_argument('--sbf-source',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--objects',type=Path,required=True);p.add_argument('--source-v2',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--train-smoke',type=int,default=0);a=p.parse_args()
    if a.train_smoke<0:raise ValueError('Invalid smoke count')
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.coordinates import MapTransform
    from sbfnav.mapdata import GROUND_LEVEL
    split='train_seen' if a.train_smoke else 'val_seen'
    ds=CityNavDataset(a.data_root,split,version='processed_citynav');n=min(a.train_smoke,len(ds)) if a.train_smoke else len(ds)
    if split=='val_seen' and n!=2470:raise ValueError(f'Expected full Revised Val-Seen 2470, found {n}')
    objects=json.loads(a.objects.read_text());a.output.mkdir(parents=True,exist_ok=True);(a.output/'rows').mkdir(exist_ok=True)
    manifest=dict(split=split,n=n,checkpoint_sha256=file_sha256(a.checkpoint),source_v2_sha256=file_sha256(a.source_v2),objects_sha256=file_sha256(a.objects),annotations_sha256=file_sha256(ds.annotation_path),observation='actual start pose only, original yaw, native384; no planned landmark views',teacher_input=False,source_sha256={x:file_sha256(Path(x)) for x in ['scripts/evaluate_unified_pgvb_v3_initial.py','graphnav/unified_pgvb_v3_runtime.py','graphnav/unified_pgvb_v3.py','graphnav/v3_diagnostics.py']})
    fp=a.output/'manifest.json'
    if fp.exists() and json.loads(fp.read_text())!=manifest:raise ValueError('Immutable evaluation protocol/source mismatch')
    atomic_json(manifest,fp);runtime=V3CausalBelief(a.source_v2,a.checkpoint,a.output/'public_inputs')
    renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'unused',image_size=384);transforms={}
    for i in range(n):
        dest=a.output/'rows'/f'{i:05d}.json'
        if dest.exists():continue
        record=ds[i];started=time.perf_counter()
        try:
            if record.map_name not in transforms:transforms[record.map_name]=MapTransform.from_raster(a.data_root/'rgbd'/f'{record.map_name}.tif',canvas_size=224,field_size=28)
            transform=transforms[record.map_name];bounds=[transform.x_min,transform.y_min,transform.x_max,transform.y_max]
            # Whitelisted public interface deliberately excludes record.object_id/references/target/trajectory.
            runtime.start(map_name=record.map_name,instruction=record.instruction,world_bounds=bounds,public_objects=objects[record.map_name])
            observation=renderer.current_rgbd(record.map_name,record.start_pose,GROUND_LEVEL[record.map_name])
            output=runtime.observe(observation=observation,pose=record.start_pose)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            # Target is introduced ONLY after the blind forward has completed.
            target=(record.target_position[:2]-np.asarray(bounds[:2]))/(np.asarray(bounds[2:])-np.asarray(bounds[:2]))
            target=torch.tensor(target,device='cuda',dtype=torch.float32)[None]
            capacity=visual_correction_capacity(output,target,runtime.bounds,runtime.model.visual_gain.tanh())[0]
            result=dict(correction_capacity=capacity,index=i,episode_id=list(record.episode_id),map_name=record.map_name,instruction=record.instruction,start_pose=record.start_pose.tolist(),public_anchor_count=len(runtime.record['landmarks']),observed=bool(output.observed_mask.any()),inference_pipeline_s=elapsed,error=None)
            for branch in ['geometry','fused']:
                met=offline_topk(output,target,runtime.bounds,branch=branch)
                result[branch]=dict(NE=met['Top1_NE_m'][0],hits={k:v[0] for k,v in met['hit_flags'].items()},top1_world=(np.asarray(bounds[:2])+np.asarray(met['Top1_xy'][0])*(np.asarray(bounds[2:])-np.asarray(bounds[:2]))).tolist())
        except Exception as e:
            if a.train_smoke:raise
            result=failed_row(i,record,e)
        atomic_json(result,dest)
        if (i+1)%25==0:print(json.dumps(dict(completed=i+1,expected=n)),flush=True)
    rows=[json.loads((a.output/'rows'/f'{i:05d}.json').read_text()) for i in range(n)]
    result=dict(coverage=dict(completed=len(rows),expected=n),split=split,errors=sum(x['error'] is not None for x in rows),metrics={})
    for branch in ['geometry','fused']:
        finite=[x[branch]['NE'] for x in rows if x[branch]['NE'] is not None]
        result['metrics'][branch]=dict(**{f'Top{k}_20m':sum(x[branch]['hits'][str(k)] for x in rows)/n for k in [1,5,20]},NE=float(np.mean(finite)) if finite else None,NE_valid_n=len(finite))
    wrong=[r['correction_capacity'] for r in rows if not r['error'] and r['correction_capacity'].get('region_defined') and not r['correction_capacity']['geometry_correct']]
    result['posthoc_capacity']=dict(geometry_wrong_n=len(wrong),no_correct_region_support=sum(not x['correct_region_observed'] for x in wrong),
        supported_but_bound_insufficient=sum(x['correct_region_observed'] and x['bound_insufficient'] for x in wrong),
        capacity_tie_undetermined=sum(x['capacity_tie_undetermined'] for x in wrong),
        capacity_sufficient_but_fusion_failed=sum(x['can_flip_under_independent_bounded_updates'] and not x['fused_correct'] for x in wrong),
        visual_region_top1_correct_but_bound_insufficient=sum(bool(x['visual_only_top1_correct_region_proxy']) and x['bound_insufficient'] for x in wrong),
        note='GT posthoc spatial proxy; optimistic independent logit changes, not visual instance correctness')
    result.update(fixes=sum(not x['geometry']['hits']['1'] and x['fused']['hits']['1'] for x in rows),breaks=sum(x['geometry']['hits']['1'] and not x['fused']['hits']['1'] for x in rows),navigation_gate_passed=(not a.train_smoke and result['errors']==0 and result['metrics']['fused']['Top1_20m']>=.55),gate_threshold=.55,closed_loop_SR_not_measured=True)
    atomic_json(result,a.output/'summary.json');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
