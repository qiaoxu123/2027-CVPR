"""Same named tasks, visual-only localization and neutral-context Oracle control."""
import argparse
from collections import Counter
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
from graphnav.visual_only_diagnostic import VisualOnlyLandmarkPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('selection','checkpoint','sbf-source','config','data-root','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-new-tasks',type=int)
    a=p.parse_args();sys.path.insert(0,str(a.sbf_source.resolve()))
    from sbfnav.config import load_config,apply_overrides
    from sbfnav.dataset import CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.stage1_runtime import GraphNavStage1Policy
    torch.set_num_threads(4)
    tasks=json.loads((a.selection/'tasks_public.json').read_text());sel=json.loads((a.selection/'manifest.json').read_text())
    if len(tasks)!=64 or sel['tasks_sha256']!=file_sha256(a.selection/'tasks_public.json'):raise ValueError('Same frozen64 required')
    cksha=file_sha256(a.checkpoint)
    if cksha!='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a':raise ValueError('Original checkpoint required')
    config=apply_overrides(load_config(a.config),['data.root='+str(a.data_root)])
    nav=config['navigation'];alt=config['altitude']
    protocol=ClosedLoopProtocol(horizon=nav['horizon'],horizontal_step_m=nav['horizontal_step_m'],vertical_step_m=nav['vertical_step_m'],
        min_altitude_m=alt['min_above_ground_m'],max_altitude_m=alt['max_above_ground_m'],stagnation_steps=nav['stagnation_steps'],success_radius_m=nav['success_distance_m'])
    if protocol!=ClosedLoopProtocol():raise ValueError('Original protocol changed')
    fingerprint=dict(tasks_sha256=sel['tasks_sha256'],checkpoint_sha256=cksha,protocol=asdict(protocol),
        methods=['E3_visual_only','E1_neutral_context'],camera_yaw='native per-pose yaw',
        zero_A3_prior=True,zero_A3_candidate_context=True,named_polygon_input_E3=False,
        sigma_oracle_m=20.,visual_gain='unchanged checkpoint',
        source_sha256={str(p):file_sha256(p) for p in [Path(__file__),Path('graphnav/visual_only_diagnostic.py'),
            Path('graphnav/landmark_navigation_diagnostic.py'),Path('graphnav/a3_runtime.py'),Path('graphnav/closedloop.py')]})
    if a.output.exists():
        if not a.resume or json.loads((a.output/'run_manifest.json').read_text())!=fingerprint:raise ValueError('Unsafe resume or overwrite')
    else:
        a.output.mkdir(parents=True);atomic_json(fingerprint,a.output/'run_manifest.json')
    path=a.output/'predictions.jsonl';rows=[json.loads(x) for x in path.open()] if path.exists() else []
    completed={(r['method'],r['index']) for r in rows}
    if len(rows)!=len(completed):raise ValueError('Duplicate prediction')
    # Catalog only configures environment transform; no target binding enters policy.
    catalog=CityReferCatalog(a.data_root/'cityrefer')
    base=PlanningStateBuilder(a.data_root,PublicCatalog(catalog),canvas_size=config['model']['canvas_size'],field_size=config['model']['field_size'])
    builder=NativeObservationBuilder(base,a.data_root/'rgbd',a.output/'observations')
    controller=BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,vertical_step_m=protocol.vertical_step_m,
        minimum_above_ground_m=protocol.min_altitude_m,maximum_above_ground_m=protocol.max_altitude_m)
    original=GraphNavStage1Policy(a.checkpoint,a.output/'unused_geometry_inputs')
    runtime=VisualOnlyLandmarkPolicy(original)
    if (runtime.stop_threshold,runtime.execution,runtime.explore_candidates)!=(.3,'guarded-blend',0):raise ValueError('Controller changed')
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();new_tasks=0
    with path.open('a') as stream:
        for t in tasks:
            if a.max_new_tasks is not None and new_tasks>=a.max_new_tasks:break
            if all((m,t['index']) in completed for m in fingerprint['methods']):continue
            for method in fingerprint['methods']:
                if (method,t['index']) in completed:continue
                torch.manual_seed(0);torch.cuda.manual_seed_all(0);np.random.seed(0)
                ep=PublicEpisode(t['map_name'],t['instruction'],np.array(t['start_pose'],float),(),())
                runtime.set_episode(ep,builder.transform(ep.map_name))
                if method=='E3_visual_only':
                    policy=runtime.predict
                else:
                    def policy(state,instruction):return runtime.predict(state,instruction,oracle_goal=t['oracle_goal_xy'])
                poses,termination,timings,diagnostics=rollout(ep,builder,controller,policy,GROUND_LEVEL[ep.map_name],protocol,synchronize=torch.cuda.synchronize)
                # Polygon is accessed only after visual-only rollout has finished.
                region=public_region(t['public_polygons'],t['world_bounds'])
                metric=polygon_trace_metrics(poses,region,termination)
                goal=shapely.distance(shapely.points(np.array([d['selected_goal_xy'] for d in diagnostics])),region)
                for d,ne in zip(diagnostics,goal):d['public_landmark_goal_distance_m']=float(ne)
                assert runtime.geometry_calls==0
                assert all(d['oracle_control']==(method=='E1_neutral_context') for d in diagnostics)
                row=dict(index=t['index'],task_id=t['task_id'],method=method,map_name=t['map_name'],category=t['category'],
                    landmark_name=t['landmark_name'],instruction=ep.instruction,poses=poses.tolist(),termination=termination,metrics=metric,
                    decision_seconds=timings,step_diagnostics=diagnostics,
                    goal_analysis=dict(initial_ne_m=float(goal[0]),final_ne_m=float(goal[-1]),ever_correct=bool((goal<=20).any()),
                        lost_correct=int(((goal[:-1]<=20)&(goal[1:]>20)).sum())))
                stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row);completed.add((method,t['index']))
                print(json.dumps(dict(method=method,index=t['index'],completed=len(rows),total=128,SR=metric['SR'],NE=metric['NE'],termination=termination,seconds=time.perf_counter()-started)),flush=True)
            new_tasks+=1
    summary={m:dict(episodes=sum(r['method']==m for r in rows),
        **{k:float(np.mean([r['metrics'][k] for r in rows if r['method']==m])) for k in ('SR','OSR','NE')},
        terminations=dict(Counter(r['termination'] for r in rows if r['method']==m))) for m in fingerprint['methods']}
    assert file_sha256(a.checkpoint)==cksha
    atomic_json(dict(complete=len(rows)==128,summary=summary,geometry_branch_calls=runtime.geometry_calls,
        seconds_current_session=time.perf_counter()-started,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        checkpoint_sha256_after=cksha),a.output/'metrics.json')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
