"""Paired real closed-loop public-landmark diagnostic, frozen original model."""
import argparse
from collections import Counter
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import time
import numpy as np
import shapely
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout
from graphnav.landmark_navigation_diagnostic import public_region,polygon_trace_metrics,oracle_goal_field


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('selection','checkpoint','sbf-source','config','data-root','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-new-tasks',type=int)
    a=p.parse_args()
    sys.path.insert(0,str(a.sbf_source.resolve()))
    from sbfnav.config import load_config,apply_overrides,validate_config
    from sbfnav.dataset import CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.stage1_runtime import GraphNavStage1Policy
    torch.set_num_threads(4)
    tasks=json.loads((a.selection/'tasks_public.json').read_text())
    selected=json.loads((a.selection/'manifest.json').read_text())
    if selected['tasks_sha256']!=file_sha256(a.selection/'tasks_public.json') or len(tasks)!=64:
        raise ValueError('Frozen64 public task set required')
    checkpoint_sha=file_sha256(a.checkpoint)
    if checkpoint_sha!='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a':
        raise ValueError('Original paper checkpoint required')
    config=apply_overrides(load_config(a.config),['data.root='+str(a.data_root)]);validate_config(config)
    nav=config['navigation'];alt=config['altitude']
    protocol=ClosedLoopProtocol(horizon=nav['horizon'],horizontal_step_m=nav['horizontal_step_m'],vertical_step_m=nav['vertical_step_m'],
        min_altitude_m=alt['min_above_ground_m'],max_altitude_m=alt['max_above_ground_m'],stagnation_steps=nav['stagnation_steps'],success_radius_m=nav['success_distance_m'])
    if protocol!=ClosedLoopProtocol():raise ValueError('Original controller protocol changed')
    fingerprint=dict(checkpoint_sha256=checkpoint_sha,selection_sha256=file_sha256(a.selection/'manifest.json'),
        tasks_sha256=selected['tasks_sha256'],protocol=asdict(protocol),camera_yaw='original per-pose native yaw; no rotation override',
        methods=['E1','E2'],goal_definition='Public named polygon union clipped to map; SR final distance<=20m, OSR ever<=20m',
        runner_sha256=file_sha256(__file__),source_sha256={k:file_sha256(Path(k)) for k in [
            'graphnav/landmark_navigation_diagnostic.py','graphnav/stage1_runtime.py','graphnav/a3_runtime.py','graphnav/closedloop.py',
            'graphnav/renderer.py','graphnav/stage1.py']},
        public_catalog_sha256=file_sha256(a.data_root/'cityrefer/objects.json'))
    if a.output.exists():
        if not a.resume or json.loads((a.output/'run_manifest.json').read_text())!=fingerprint:
            raise ValueError('Refuse overwrite or resume under changed inference inputs')
    else:
        a.output.mkdir(parents=True);atomic_json(fingerprint,a.output/'run_manifest.json')
    path=a.output/'predictions.jsonl'
    rows=[json.loads(line) for line in path.open()] if path.exists() else []
    completed={(r['method'],r['index']) for r in rows}
    if len(completed)!=len(rows):raise ValueError('Duplicate completed task')
    catalog=CityReferCatalog(a.data_root/'cityrefer')
    builder=PlanningStateBuilder(a.data_root,PublicCatalog(catalog),canvas_size=config['model']['canvas_size'],field_size=config['model']['field_size'],
        max_height_above_ground_m=config['state']['max_height_above_ground_m'],trajectory_radius_m=config['state']['trajectory_radius_m'])
    builder=NativeObservationBuilder(builder,a.data_root/'rgbd',a.output/'observations')
    controller=BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,vertical_step_m=protocol.vertical_step_m,
        minimum_above_ground_m=protocol.min_altitude_m,maximum_above_ground_m=protocol.max_altitude_m)
    runtime=GraphNavStage1Policy(a.checkpoint,a.output/'public_inputs')
    if (runtime.stop_threshold,runtime.execution,runtime.explore_candidates,runtime.stop_positive_weight)!=(.3,'guarded-blend',0,20.):
        raise ValueError('Original learned controller/Stop changed')
    assert all(not p.requires_grad for p in runtime.model.parameters())
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();new_tasks=0
    with path.open('a') as stream:
        for task in tasks:
            if a.max_new_tasks is not None and new_tasks>=a.max_new_tasks:break
            if all((m,task['index']) in completed for m in ('E1','E2')):continue
            for method in ('E1','E2'):
                if (method,task['index']) in completed:continue
                torch.manual_seed(0);np.random.seed(0);torch.cuda.manual_seed_all(0)
                # Only new landmark name, start and public references enter policy.
                episode=PublicEpisode(task['map_name'],task['instruction'],np.asarray(task['start_pose'],float),(),(task['landmark_name'],))
                transform=builder.transform(episode.map_name)
                runtime.set_episode(episode,transform,catalog)
                region=public_region(task['public_polygons'],task['world_bounds'])
                hook=None;raw_goals=[]
                if method=='E1':
                    def force_public_goal(module,args,output):
                        field=output.belief
                        xy=runtime.bounds[:,:2]+field.candidate_xy[:,0]*(runtime.bounds[:,2:]-runtime.bounds[:,:2])
                        raw_goals.append(xy[0].float().cpu().tolist())
                        oracle=oracle_goal_field(field,runtime.inputs['grid_xy'],runtime.bounds,task['oracle_goal_xy'],diagnostic=True)
                        return replace(output,belief=oracle)
                    hook=runtime.model.register_forward_hook(force_public_goal)
                try:
                    poses,termination,timings,diagnostics=rollout(episode,builder,controller,runtime.predict,
                        GROUND_LEVEL[episode.map_name],protocol,synchronize=torch.cuda.synchronize)
                finally:
                    if hook is not None:hook.remove()
                metrics=polygon_trace_metrics(poses,region,termination)
                goals=np.asarray([d['selected_goal_xy'] for d in diagnostics])
                goal_distance=shapely.distance(shapely.points(goals),region)
                for step,(d,gne) in enumerate(zip(diagnostics,goal_distance)):
                    d['public_landmark_goal_distance_m']=float(gne)
                    if method=='E1':d['pre_oracle_model_goal_xy']=raw_goals[step]
                row=dict(index=task['index'],task_id=task['task_id'],method=method,map_name=task['map_name'],
                    category=task['category'],public_types=task['public_types'],landmark_name=task['landmark_name'],
                    instruction=task['instruction'],poses=poses.tolist(),termination=termination,metrics=metrics,
                    decision_seconds=timings,step_diagnostics=diagnostics,
                    goal_analysis=dict(initial_ne_m=float(goal_distance[0]),final_ne_m=float(goal_distance[-1]),
                        ever_correct=bool((goal_distance<=20).any()),
                        lost_correct=int(((goal_distance[:-1]<=20)&(goal_distance[1:]>20)).sum())))
                stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row);completed.add((method,task['index']))
                print(json.dumps(dict(method=method,index=task['index'],completed=len(rows),total=128,SR=metrics['SR'],NE=metrics['NE'],
                    termination=termination,seconds=time.perf_counter()-started)),flush=True)
            new_tasks+=1
    summary={}
    for method in ('E1','E2'):
        group=[r for r in rows if r['method']==method]
        if group:
            summary[method]=dict(episodes=len(group),successes=sum(r['metrics']['SR'] for r in group),
                **{k:float(np.mean([r['metrics'][k] for r in group])) for k in ('SR','OSR','NE')},
                terminations=dict(Counter(r['termination'] for r in group)))
    atomic_json(dict(complete=len(rows)==128,summary=summary,seconds_current_session=time.perf_counter()-started,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        checkpoint_sha256_after=file_sha256(a.checkpoint)),a.output/'metrics.json')
    if file_sha256(a.checkpoint)!=checkpoint_sha:raise ValueError('Checkpoint unexpectedly changed')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
