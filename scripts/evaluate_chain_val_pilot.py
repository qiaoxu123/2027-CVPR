"""Fixed frozen Val-Seen128 paired pilot; no TRAIN fitting or Test access.

Reuses original rollout, metrics, renderer, controller and Stop unchanged.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import ClosedLoopProtocol,PublicEpisode,PublicCatalog,rollout,score_trace,aggregate
from graphnav.fixed_yaw_observation import FixedYawObservation


def validate_val_pilot(frozen,episode_count):
    if frozen.get('split')!='val_seen' or frozen.get('purpose')!='fixed_val_seen_128' or frozen.get('no_parameter_selection') is not True:
        raise ValueError('explicit fixed Val-Seen diagnostic freeze required')
    ids=frozen.get('indices',[])
    if len(ids)!=128 or len(set(ids))!=128 or any(type(i) is not int or i<0 or i>=episode_count for i in ids):
        raise ValueError('exactly128 valid distinct episodes required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('sbf-source','data-root','config','checkpoint','local-checkpoint','public-roles','episode-manifest','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=('original','real'),default=['original','real'])
    p.add_argument('--resume',action='store_true')
    p.set_defaults(field_mode='grid',smoke_episodes=None)
    a=p.parse_args()
    if a.smoke_episodes is not None and a.smoke_episodes<1:raise ValueError('positive smoke episode count required')
    if len(a.methods)!=2 or set(a.methods)!={'original','real'}:raise ValueError('paired Original/Ours only')
    sys.path.insert(0,str(a.sbf_source.resolve()))
    from sbfnav.config import load_config,apply_overrides,validate_config
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.stage1_runtime import GraphNavStage1Policy
    from graphnav.local_chain_runtime import LocalChainPolicy
    torch.set_num_threads(4)
    config=apply_overrides(load_config(a.config),['data.root='+str(a.data_root)]);validate_config(config)
    nav=config['navigation'];alt=config['altitude']
    protocol=ClosedLoopProtocol(horizon=nav['horizon'],horizontal_step_m=nav['horizontal_step_m'],vertical_step_m=nav['vertical_step_m'],
        min_altitude_m=alt['min_above_ground_m'],max_altitude_m=alt['max_above_ground_m'],stagnation_steps=nav['stagnation_steps'],success_radius_m=nav['success_distance_m'])
    if protocol!=ClosedLoopProtocol():raise ValueError('original frozen controller protocol required')
    dataset=CityNavDataset(a.data_root,'val_seen',version='processed_citynav')
    catalog=CityReferCatalog(a.data_root/'cityrefer')
    frozen=json.loads(a.episode_manifest.read_text())
    validate_val_pilot(frozen,len(dataset))
    if frozen.get('pilot_runner_sha256')!=file_sha256(__file__):raise ValueError('frozen pilot runner source changed')
    for name,expected in frozen.get('source_sha256',{}).items():
        if file_sha256(Path(__file__).resolve().parents[1]/name)!=expected:raise ValueError('frozen inference source changed: '+name)
    if frozen['split']!='val_seen' or frozen['checkpoint_sha256']!=file_sha256(a.checkpoint) or frozen['local_checkpoint_sha256']!=file_sha256(a.local_checkpoint):
        raise ValueError('episode/model freeze mismatch')
    if frozen['annotation_sha256']!=file_sha256(dataset.annotation_path) or frozen['public_roles_sha256']!=file_sha256(a.public_roles):
        raise ValueError('frozen text/data mismatch')
    indices=frozen['indices'][:a.smoke_episodes] if a.smoke_episodes else frozen['indices']
    fingerprint=dict(source_sha256=frozen.get('source_sha256',{}),split='val_seen',dataset_version='processed_citynav',purpose='fixed_val_seen_128',episode_manifest_sha256=file_sha256(a.episode_manifest),checkpoint_sha256=file_sha256(a.checkpoint),
                     local_checkpoint_sha256=file_sha256(a.local_checkpoint),public_roles_sha256=file_sha256(a.public_roles),
                     controller_protocol=asdict(protocol),camera_yaw_rad=0.,methods=a.methods,smoke_episodes=a.smoke_episodes,field_mode=a.field_mode,
                     raster_sha256=file_sha256(Path(__file__).resolve().parents[1]/'graphnav/local_chain_raster.py'),
                     runner_sha256=file_sha256(__file__),runtime_sha256=file_sha256(Path(__file__).resolve().parents[1]/'graphnav/local_chain_runtime.py'),
                     public_catalog_sha256=file_sha256(a.data_root/'cityrefer/objects.json'),
                     annotation_sha256=file_sha256(dataset.annotation_path))
    if a.output.exists():
        if not a.resume:raise FileExistsError('preserve prior navigation')
        if json.loads((a.output/'run_manifest.json').read_text())!=fingerprint:raise ValueError('resume protocol/source changed')
    else:
        a.output.mkdir(parents=True);atomic_json(fingerprint,a.output/'run_manifest.json')
    summaries={}
    for method in a.methods:
        out=a.output/method;out.mkdir(exist_ok=True)
        existing=[];path=out/'predictions.jsonl'
        if path.exists():existing=[json.loads(line) for line in path.read_text().splitlines()]
        completed={r['index'] for r in existing}
        if not completed.issubset(set(indices)) or len(completed)!=len(existing):raise ValueError('mismatched/duplicate saved episodes')
        rows=[r['metrics'] for r in existing]
        if len(existing)==len(indices):
            summaries[method]=json.loads((out/'metrics.json').read_text());continue
        torch.manual_seed(0);np.random.seed(0);torch.cuda.manual_seed_all(0)
        builder=PlanningStateBuilder(a.data_root,PublicCatalog(catalog),canvas_size=config['model']['canvas_size'],field_size=config['model']['field_size'],
                max_height_above_ground_m=config['state']['max_height_above_ground_m'],trajectory_radius_m=config['state']['trajectory_radius_m'])
        builder=NativeObservationBuilder(builder,a.data_root/'rgbd',out/'observations')
        builder.renderer=FixedYawObservation(builder.renderer,0.)
        controller=BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,vertical_step_m=protocol.vertical_step_m,
                        minimum_above_ground_m=protocol.min_altitude_m,maximum_above_ground_m=protocol.max_altitude_m)
        if method=='original':runtime=GraphNavStage1Policy(a.checkpoint,out/'public_inputs')
        else:runtime=LocalChainPolicy(a.checkpoint,a.local_checkpoint,a.public_roles,out/'public_inputs',control=method,field_mode=a.field_mode)
        if (runtime.stop_threshold,runtime.execution,runtime.explore_candidates,runtime.stop_positive_weight)!=(.3,'guarded-blend',0,20.):
            raise ValueError('control/Stop changed')
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        def observed_policy(state,instruction):
            prediction=runtime.predict(state,instruction)
            obs=state.current_observation
            prediction.diagnostics.update(observed_rgb_sha256=runtime.native.image_key(obs['bgr']),
                camera_pose=np.asarray(obs['pose']).tolist(),footprint_world=np.asarray(obs['footprint_world']).tolist())
            return prediction
        with path.open('a') as stream:
            for index in indices:
                if index in completed:continue
                record=dataset[index]
                episode=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),
                    tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
                runtime.set_episode(episode,builder.transform(episode.map_name),catalog)
                poses,termination,timings,diagnostics=rollout(episode,builder,controller,observed_policy,
                    GROUND_LEVEL[episode.map_name],protocol,synchronize=torch.cuda.synchronize)
                # Policy has finished. Only the scorer sees target/reference labels.
                metrics=score_trace(poses,record.target_position,record.trajectory,termination,radius=20.,decision_seconds=timings)
                goals=np.asarray([d['selected_goal_xy'] for d in diagnostics])
                goal_distance=np.linalg.norm(goals-record.target_position[:2],axis=-1)
                row=dict(index=index,episode_id=list(record.episode_id),map_name=record.map_name,poses=poses.tolist(),
                    termination=termination,metrics=metrics,decision_seconds=timings,step_diagnostics=diagnostics,
                    posthoc_goal_analysis=dict(first_ne_m=float(goal_distance[0]),final_ne_m=float(goal_distance[-1]),
                        ever_correct=bool((goal_distance<=20).any()),lost_correct_goal=int(((goal_distance[:-1]<=20)&(goal_distance[1:]>20)).sum()),
                        recovered_goal=int(((goal_distance[:-1]>20)&(goal_distance[1:]<=20)).sum())))
                stream.write(json.dumps(row)+'\n');stream.flush();rows.append(metrics);existing.append(row)
                print(json.dumps(dict(method=method,completed=len(rows),total=len(indices),seconds=time.perf_counter()-started)),flush=True)
        report=dict(method=method,split='val_seen',episodes=len(indices),metrics=aggregate(rows),
            terminations=dict(Counter(r['termination'] for r in existing)),seconds_current_session=time.perf_counter()-started,
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,protocol=asdict(protocol),
            lost_correct_goals=sum(r['posthoc_goal_analysis']['lost_correct_goal'] for r in existing),
            warning='Fixed128 Val-Seen pilot, NOT full split SR or model selection; no Test access.')
        atomic_json(report,out/'metrics.json');summaries[method]=report
        del runtime,builder;gc.collect();torch.cuda.empty_cache()
    atomic_json(summaries,a.output/'comparison.json')
    print(json.dumps({m:r['metrics'] for m,r in summaries.items()},indent=2))


if __name__=='__main__':main()
