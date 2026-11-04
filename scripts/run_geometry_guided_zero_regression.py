"""Frozen64 RGB-zero causal control; no model/parameter selection."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import shapely
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout
from graphnav.landmark_navigation_diagnostic import public_region,polygon_trace_metrics
from graphnav.fixed_yaw_observation import FixedYawObservation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('selection','checkpoint','visual-checkpoint','sbf-source','config','data-root','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();sys.path.insert(0,str(a.sbf_source))
    from sbfnav.config import load_config,apply_overrides
    from sbfnav.dataset import CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.geometry_guided_zero_runtime import ZeroFeatureLandmarkPolicy as GeometryGuidedLandmarkPolicy
    torch.set_num_threads(4)
    tasks=json.loads((a.selection/'tasks_public.json').read_text());proof=json.loads((a.selection/'manifest.json').read_text())
    sealed=json.loads((a.visual_checkpoint.parent/'freeze.json').read_text())
    if len(tasks)!=64 or file_sha256(a.visual_checkpoint)!=sealed['checkpoint_sha256'] or file_sha256(a.selection/'tasks_public.json')!=proof['tasks_sha256']:raise ValueError('Frozen model/task manifest mismatch')
    cksha=file_sha256(a.checkpoint)
    if cksha!='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a':raise ValueError('Original geometry checkpoint required')
    config=apply_overrides(load_config(a.config),['data.root='+str(a.data_root)]);nav=config['navigation'];alt=config['altitude']
    protocol=ClosedLoopProtocol(horizon=nav['horizon'],horizontal_step_m=nav['horizontal_step_m'],vertical_step_m=nav['vertical_step_m'],
        min_altitude_m=alt['min_above_ground_m'],max_altitude_m=alt['max_above_ground_m'],stagnation_steps=nav['stagnation_steps'],success_radius_m=nav['success_distance_m'])
    if protocol!=ClosedLoopProtocol():raise ValueError('Controller protocol changed')
    fingerprint=dict(checkpoint_sha256=cksha,visual_checkpoint_sha256=sealed['checkpoint_sha256'],tasks_sha256=proof['tasks_sha256'],
        protocol=asdict(protocol),camera_yaw_rad=0.,methods=['GeometryZeroVisual'],
        no_val_parameter_selection=True,source_sha256={str(p):file_sha256(p) for p in [Path(__file__),Path('graphnav/geometry_guided_zero_runtime.py'),Path('graphnav/geometry_guided_runtime.py'),
            Path('graphnav/geometry_guided_visual.py'),Path('graphnav/a3_runtime.py'),Path('graphnav/closedloop.py')]})
    if a.output.exists():
        if not a.resume or json.loads((a.output/'run_manifest.json').read_text())!=fingerprint:raise ValueError('Unsafe resume')
    else:a.output.mkdir(parents=True);atomic_json(fingerprint,a.output/'run_manifest.json')
    path=a.output/'predictions.jsonl';rows=[json.loads(x) for x in path.open()] if path.exists() else [];completed={(r['method'],r['index']) for r in rows}
    catalog=CityReferCatalog(a.data_root/'cityrefer')
    base=PlanningStateBuilder(a.data_root,PublicCatalog(catalog),canvas_size=config['model']['canvas_size'],field_size=config['model']['field_size'])
    builder=NativeObservationBuilder(base,a.data_root/'rgbd',a.output/'observations');builder.renderer=FixedYawObservation(builder.renderer,0.)
    controller=BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,vertical_step_m=protocol.vertical_step_m,
        minimum_above_ground_m=protocol.min_altitude_m,maximum_above_ground_m=protocol.max_altitude_m)
    runtime=GeometryGuidedLandmarkPolicy(a.checkpoint,a.visual_checkpoint,a.output/'public_inputs')
    if (runtime.stop_threshold,runtime.execution,runtime.stop_positive_weight)!=(.3,'guarded-blend',20.):raise ValueError('Control changed')
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with path.open('a') as stream:
        for task in tasks:
            for method in fingerprint['methods']:
                if (method,task['index']) in completed:continue
                torch.manual_seed(0);np.random.seed(0);torch.cuda.manual_seed_all(0)
                episode=PublicEpisode(task['map_name'],task['instruction'],np.array(task['start_pose'],float),(),(task['landmark_name'],))
                runtime.set_episode(episode,builder.transform(episode.map_name),catalog)
                def policy(state,text):return runtime.predict(state,text,use_visual=True)
                poses,termination,timings,diagnostics=rollout(episode,builder,controller,policy,GROUND_LEVEL[episode.map_name],protocol,synchronize=torch.cuda.synchronize)
                region=public_region(task['public_polygons'],task['world_bounds']);metrics=polygon_trace_metrics(poses,region,termination)
                distances=shapely.distance(shapely.points(np.array([d['selected_goal_xy'] for d in diagnostics])),region)
                row=dict(index=task['index'],method=method,task_id=task['task_id'],map_name=task['map_name'],category=task['category'],
                    landmark_name=task['landmark_name'],instruction=task['instruction'],poses=poses.tolist(),metrics=metrics,termination=termination,
                    decision_seconds=timings,step_diagnostics=diagnostics,goal_analysis=dict(initial_ne_m=float(distances[0]),final_ne_m=float(distances[-1]),
                        ever_correct=bool((distances<=20).any()),lost_correct=int(((distances[:-1]<=20)&(distances[1:]>20)).sum())))
                stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
                print(json.dumps(dict(index=task['index'],method=method,completed=len(rows),SR=metrics['SR'],NE=metrics['NE'],seconds=time.perf_counter()-started)),flush=True)
    results={m:dict(episodes=sum(r['method']==m for r in rows),**{k:float(np.mean([r['metrics'][k] for r in rows if r['method']==m])) for k in ('SR','OSR','NE')}) for m in fingerprint['methods']}
    assert file_sha256(a.checkpoint)==cksha
    atomic_json(dict(complete=len(rows)==64,results=results,seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        original_checkpoint_sha256_after=cksha,visual_checkpoint_sha256_after=file_sha256(a.visual_checkpoint)),a.output/'metrics.json')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
