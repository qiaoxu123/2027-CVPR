"""Frozen original-target Val-Seen rollout; all eligible tasks stay in denominator."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback
import cv2
import numpy as np
import torch
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.closedloop import PublicEpisode, PublicCatalog, ClosedLoopProtocol, rollout, score_trace


def main():
    p=argparse.ArgumentParser()
    for k in ('selection','stage1','original-v2','checkpoint','sbf-source','config','data-root','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--keep-going-on-error',action='store_true',
                   help='Formal diagnostic: retain failed episodes in denominator and finish all tasks')
    a=p.parse_args();sys.path.insert(0,str(a.sbf_source))
    from sbfnav.config import load_config, apply_overrides, validate_config
    from sbfnav.dataset import CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.direct_target_runtime import UnifiedPGVBDirectTargetPolicy
    torch.set_num_threads(4)
    tasks=json.loads((a.selection/'tasks_public.json').read_text())
    selection=json.loads((a.selection/'selection_manifest.json').read_text())
    labels=json.loads((a.selection/'labels_private.json').read_text())
    order=selection['pilot_indices']+[t['index'] for t in tasks if t['index'] not in selection['pilot_indices']]
    if a.pilot:order=selection['pilot_indices']
    config=apply_overrides(load_config(a.config),['data.root='+str(a.data_root)]);validate_config(config)
    nav=config['navigation'];alt=config['altitude']
    protocol=ClosedLoopProtocol(horizon=nav['horizon'],horizontal_step_m=nav['horizontal_step_m'],
        vertical_step_m=nav['vertical_step_m'],min_altitude_m=alt['min_above_ground_m'],
        max_altitude_m=alt['max_above_ground_m'],stagnation_steps=nav['stagnation_steps'],success_radius_m=nav['success_distance_m'])
    if protocol!=ClosedLoopProtocol():raise ValueError('Original evaluation/control protocol changed')
    manifest=dict(checkpoint_sha256=file_sha256(a.checkpoint),stage1_sha256=file_sha256(a.stage1),
        original_v2_sha256=file_sha256(a.original_v2),selection_sha256=file_sha256(a.selection/'tasks_public.json'),
        protocol=asdict(protocol),methods=['Geometry','Relation','RGB'],eligible_tasks=len(tasks),
        keep_going_on_error=a.keep_going_on_error,
        pilot_indices=selection['pilot_indices'],GT_online=False,RGB_yaw='unchanged actual pose',
        code_sha256={k:file_sha256(k) for k in [__file__,'graphnav/direct_target_runtime.py',
            'graphnav/direct_target.py','graphnav/unified_pgvb.py','graphnav/pgvb_observation.py',
            'graphnav/stable_navigation_goal.py','graphnav/trajectory_policy.py','graphnav/a3_runtime.py']})
    a.output.mkdir(parents=True,exist_ok=True);mp=a.output/'freeze_manifest.json'
    if mp.exists():
        if not a.resume or json.loads(mp.read_text())!=manifest:raise ValueError('Frozen evaluation inputs changed')
    else:atomic_json(manifest,mp)
    path=a.output/'predictions.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    completed={(r['index'],r['method']) for r in rows}
    if len(completed)!=len(rows):raise ValueError('Duplicate results')
    catalog=CityReferCatalog(a.data_root/'cityrefer')
    builder=PlanningStateBuilder(a.data_root,PublicCatalog(catalog),canvas_size=config['model']['canvas_size'],
        field_size=config['model']['field_size'],max_height_above_ground_m=config['state']['max_height_above_ground_m'],
        trajectory_radius_m=config['state']['trajectory_radius_m'])
    builder=NativeObservationBuilder(builder,a.data_root/'rgbd',a.output/'observations')
    controller=BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,vertical_step_m=protocol.vertical_step_m,
        minimum_above_ground_m=protocol.min_altitude_m,maximum_above_ground_m=protocol.max_altitude_m)
    runtime=UnifiedPGVBDirectTargetPolicy(a.stage1,a.original_v2,a.checkpoint,a.output/'public_inputs')
    assert (runtime.stop_threshold,runtime.execution,runtime.explore_candidates,runtime.stop_positive_weight)==(.3,'guarded-blend',0,20.)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with path.open('a') as stream:
        for index in order:
            task=tasks[index]
            for method in manifest['methods']:
                if (index,method) in completed:continue
                torch.manual_seed(42);np.random.seed(42);torch.cuda.manual_seed_all(42)
                episode=PublicEpisode(task['map_name'],task['instruction'],np.asarray(task['start_pose'],float),(),
                    tuple(q['landmark']['name'] for q in task['anchors']))
                runtime.method=method;captures=[];directory=a.output/'decisions'/f'{index:04d}_{method}';directory.mkdir(parents=True,exist_ok=True)
                def capture(state,result,field):
                    step=len(captures);file=directory/f'{step:02d}.npz'
                    payload=dict(pose=state.pose,footprint=state.current_observation['footprint_world'],
                        geometry=result['geometry_field'].belief_logits[0].float().cpu().numpy(),
                        relation=result['relation_field'].belief_logits[0].float().cpu().numpy(),
                        full=result['field'].belief_logits[0].float().cpu().numpy(),
                        selected_ids=field.candidate_ids[0].cpu().numpy(),coverage=result['coverage'][0].cpu().numpy())
                    for key in ('rejected_relation_field','rejected_visual_field'):
                        if key in result:
                            payload[key]=result[key].belief_logits[0].float().cpu().numpy()
                    np.savez_compressed(file,**payload)
                    cv2.imwrite(str(directory/f'{step:02d}.png'),state.current_observation['bgr'])
                    captures.append(str(file))
                runtime.direct_audit_callback=capture
                error=None;timings=[];diagnostics=[];poses=[episode.start_pose]
                try:
                    runtime.set_direct_episode(episode,builder.transform(episode.map_name),catalog,task)
                    poses,termination,timings,diagnostics=rollout(episode,builder,controller,runtime.predict,
                        GROUND_LEVEL[episode.map_name],protocol,synchronize=torch.cuda.synchronize)
                except Exception:
                    error=traceback.format_exc();termination='inference_error'
                # Private labels accessed strictly after the real rollout.
                label=labels[index]
                metrics=score_trace(poses,label['target'],label['reference'],termination,decision_seconds=timings)
                if error:metrics.update(SR=0.,SPL=0.)
                goals=[d.get('selected_goal_xy') for d in diagnostics]
                row=dict(index=index,episode=task['episode'],map_name=task['map_name'],instruction=task['instruction'],method=method,
                    metrics=metrics,termination=termination,error=error,poses=np.asarray(poses).tolist(),diagnostics=diagnostics,
                    goals=goals,captures=captures,visual_updates=sum(bool(d.get('visual_active')) for d in diagnostics),
                    peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
                stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row);completed.add((index,method))
                print(json.dumps(dict(index=index,method=method,termination=termination,**metrics)),flush=True)
                if error and not a.keep_going_on_error:
                    # Stop on the first interface failure. It is retained as a
                    # failure; fix evidence before spending GPU on the full set.
                    raise RuntimeError(error)
    summary={}
    for group,indices in [('pilot',set(selection['pilot_indices'])),('all',set(t['index'] for t in tasks))]:
        summary[group]={}
        for method in manifest['methods']:
            selected=[r for r in rows if r['index'] in indices and r['method']==method]
            summary[group][method]=dict(episodes=len(selected),expected=len(indices),
                **{k:float(np.mean([r['metrics'][k] for r in selected])) for k in ('SR','OSR','SPL','NE','nDTW','steps','inference_ms')},
                errors=sum(r['error'] is not None for r in selected),
                policy_stop=sum(r['termination']=='policy_stop' for r in selected),
                horizon=sum(r['termination']=='horizon' for r in selected))
    summary['elapsed_seconds']=time.perf_counter()-started
    atomic_json(summary,a.output/('pilot_summary.json' if a.pilot else 'full_summary.json'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
