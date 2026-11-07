"""Paired full Refined Val-Seen navigation, frozen V2 Geometry and original controller.
Uses the same blind public-name binding as evaluate_v2_public_candidate_recall.
No OSM, RGB fusion, spatial-NMS goal substitution, or privileged goal selection.
"""
import argparse,json,sys,time
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import torch
from graphnav.a3_runtime import A3TrajectoryPolicy
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout,score_trace,aggregate
from graphnav.dynamic_runtime import NativeObservationBuilder
from graphnav.token_language import FrozenBERTTokens
from graphnav.trajectory_policy import CandidateTrajectoryStop
from scripts.unified_pgvb_v3_pilot import _load_base
from scripts.analyze_instruction_landmark_association import PublicNames


class FrozenV2GeometryPolicy(A3TrajectoryPolicy):
    def __init__(self,source,trained,stage1,directory,arm):
        self.device=torch.device('cuda');self.scratch=Path(directory);self.scratch.mkdir(parents=True,exist_ok=True)
        self.model,self.args=_load_base(source,self.device)
        if arm=='trained_v2':
            payload=torch.load(trained,map_location='cpu',weights_only=True)
            self.model.load_state_dict({k[len('geometry.'):]:v for k,v in payload['model'].items() if k.startswith('geometry.')},strict=True)
        self.model.eval().requires_grad_(False)
        source_state=torch.load(stage1,map_location='cpu',weights_only=True)
        if source_state['protocol']['name']!='graphnav-v1-stage1' or source_state['protocol']['active_observation']:raise ValueError('Original non-active controller required')
        self.head=CandidateTrajectoryStop(hidden=self.args.hidden).to(self.device).eval().requires_grad_(False)
        self.head.load_state_dict({k[len('trajectory.'):]:v for k,v in source_state['model'].items() if k.startswith('trajectory.')},strict=True)
        self.encoder=FrozenBERTTokens(self.args.text_backbone,device=self.device,max_length=getattr(self.args,'max_text_length',96))
        expected=source_state.get('config',{}).get('frozen_backbones',{}).get('bert_revision')
        if expected and self.encoder.revision!=expected:raise ValueError('Frozen BERT revision mismatch')
        self.cache={};self.field_cache={};self.record=None
        self.stop_threshold=.3;self.stop_positive_weight=20.;self.execution='guarded-blend';self.explore_candidates=0


def main():
    p=argparse.ArgumentParser()
    for k in ['source-v2','trained-v2','stage1','data','sbf-source','config','initial-eval','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--max-new-per-arm',type=int,default=0,help='Engineering smoke only; zero runs all episodes')
    p.add_argument('--arms',nargs='+',choices=['original_v2','trained_v2'],default=['original_v2','trained_v2'])
    a=p.parse_args();sys.path.insert(0,str(a.sbf_source))
    from sbfnav.config import load_config,apply_overrides,validate_config
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
    config=apply_overrides(load_config(a.config),['data.root='+str(a.data)]);validate_config(config)
    nav,alt=config['navigation'],config['altitude'];protocol=ClosedLoopProtocol(horizon=nav['horizon'],horizontal_step_m=nav['horizontal_step_m'],vertical_step_m=nav['vertical_step_m'],min_altitude_m=alt['min_above_ground_m'],max_altitude_m=alt['max_above_ground_m'],stagnation_steps=nav['stagnation_steps'],success_radius_m=nav['success_distance_m'])
    if protocol!=ClosedLoopProtocol():raise ValueError('Original protocol changed')
    ds=CityNavDataset(a.data,'val_seen',version='processed_citynav');assert len(ds)==2470
    initial_manifest=json.load(open(a.initial_eval/'summary.json'))
    for k,path in [('annotation_sha256',ds.annotation_path),('catalog_sha256',a.data/'cityrefer/objects.json'),('source_v2_sha256',a.source_v2),('trained_v2_sha256',a.trained_v2)]:
        if file_sha256(path)!=initial_manifest[k]:raise ValueError('Input differs from candidate recall: '+k)
    if file_sha256(a.stage1)!='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a':raise ValueError('Original controller source mismatch')
    catalog=CityReferCatalog(a.data/'cityrefer');names=PublicNames(catalog.objects)
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(seed=42,expected=2470,split='val_seen',protocol=asdict(protocol),stop_threshold=.3,arrival_guard_m=.001,execution='guarded-blend',explore_candidates=0,binding='Blind raw instruction + public names, identical to preceding candidate recall',OSM=False,visual_fusion=False,spatial_nms_changes_to_navigation=False,checkpoint_shas={k:initial_manifest[k] for k in ['source_v2_sha256','trained_v2_sha256']},stage1_sha256=file_sha256(a.stage1),annotations_sha256=file_sha256(ds.annotation_path),catalog_sha256=initial_manifest['catalog_sha256'],source_sha256={k:file_sha256(Path(k)) for k in ['scripts/evaluate_v2_public_navigation.py','graphnav/a3_runtime.py','graphnav/closedloop.py','graphnav/candidate_geometry.py','graphnav/renderer.py']})
    dest=a.output/'manifest.json'
    if dest.exists() and json.load(open(dest))!=manifest:raise ValueError('Refuse changed input/protocol on resume')
    atomic_json(manifest,dest)
    for arm in a.arms:
        armout=a.output/arm;(armout/'rows').mkdir(parents=True,exist_ok=True)
        b=PlanningStateBuilder(a.data,PublicCatalog(catalog),canvas_size=config['model']['canvas_size'],field_size=config['model']['field_size'],max_height_above_ground_m=config['state']['max_height_above_ground_m'],trajectory_radius_m=config['state']['trajectory_radius_m'])
        builder=NativeObservationBuilder(b,a.data/'rgbd',a.output/'observations')
        controller=BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,vertical_step_m=protocol.vertical_step_m,minimum_above_ground_m=protocol.min_altitude_m,maximum_above_ground_m=protocol.max_altitude_m)
        runtime=FrozenV2GeometryPolicy(a.source_v2,a.trained_v2,a.stage1,armout/'public_inputs',arm)
        public_fields=np.load(a.initial_eval/(arm+'_public_fields.npz'))
        new=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for i in range(len(ds)):
            path=armout/'rows'/f'{i:05d}.json'
            if path.exists():continue
            if a.max_new_per_arm and new>=a.max_new_per_arm:break
            record=ds[i];refnames=tuple(dict.fromkeys(n for m in names.match(record.map_name,record.instruction) for n in m['public_names']))
            public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),(),refnames)
            runtime.set_episode(public,builder.transform(record.map_name),catalog)
            # Freeze the same raw Geometry candidates as the offline experiment; no NMS/GT selection.
            field,_=runtime.frozen_field();bounds=np.asarray(runtime.record['world_bounds']);world=bounds[:2]+field.candidate_xy[0,0].float().cpu().numpy()*(bounds[2:]-bounds[:2]);baseline=public_fields['xy'][i,int(torch.tensor(public_fields['logits'][i]).topk(int(runtime.args.top_k)).indices[0])];agreement=float(np.linalg.norm(world-baseline))
            t=time.perf_counter();poses,termination,timings,diagnostics=rollout(public,builder,controller,runtime.predict,GROUND_LEVEL[record.map_name],protocol,synchronize=torch.cuda.synchronize)
            # Private labels/reference trajectory first used AFTER rollout, only for scoring.
            metrics=score_trace(poses,record.target_position,record.trajectory,termination,radius=protocol.success_radius_m,decision_seconds=timings)
            atomic_json(dict(index=i,episode_id=list(record.episode_id),map_name=record.map_name,instruction=record.instruction,public_reference_names=list(refnames),poses=poses.tolist(),termination=termination,metrics=metrics,diagnostics=diagnostics,initial_top1_offline_difference_m=agreement,rollout_wall_s=time.perf_counter()-t),path)
            new+=1
            if new%10==0 or new<=2:print(json.dumps(dict(arm=arm,new=new,index=i,SR=metrics['SR'],NE=metrics['NE'],steps=metrics['steps'],elapsed_s=time.perf_counter()-started)),flush=True)
            if new%25==0:
                rr=[json.load(open(x)) for x in sorted((armout/'rows').glob('*.json'))];atomic_json(dict(completed=len(rr),expected=2470,partial=True,metrics=aggregate([r['metrics'] for r in rr])),armout/'progress.json')
        rows=[json.load(open(x)) for x in sorted((armout/'rows').glob('*.json'))]
        complete=len(rows)==2470
        atomic_json(dict(complete=complete,coverage=dict(completed=len(rows),expected=2470),metrics=aggregate([r['metrics'] for r in rows]),elapsed_new_s=time.perf_counter()-started,gpu_peak_GiB=torch.cuda.max_memory_allocated()/2**30,offline_top1_agreement_within_1m=sum(r['initial_top1_offline_difference_m']<=1 for r in rows),termination_counts={x:sum(r['termination']==x for r in rows) for x in ['policy_stop','horizon','stagnation']}),armout/('metrics.json' if complete else 'smoke_metrics.json'))
        del runtime,builder,controller;torch.cuda.empty_cache()
    print('Requested pass complete; full metrics only exist for 2470/2470.',flush=True)
if __name__=='__main__':main()
