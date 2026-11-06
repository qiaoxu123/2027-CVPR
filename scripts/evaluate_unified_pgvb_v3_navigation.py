"""Full Val-Seen V3 navigation; locked behind full initial Top1 >=55%, no Test."""
import argparse,json,sys,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout,score_trace,aggregate
from graphnav.unified_pgvb_v3_runtime import V3NavigationPolicy
from graphnav.dynamic_runtime import NativeObservationBuilder

def main():
    p=argparse.ArgumentParser()
    for k in ['sbf-source','config','data-root','source-v2','checkpoint','stage1','initial-eval','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--geometry-only',action='store_true');a=p.parse_args()
    gate=json.loads((a.initial_eval/'summary.json').read_text());gm=json.loads((a.initial_eval/'manifest.json').read_text())
    if not gate.get('navigation_gate_passed') or gate['errors'] or gate['coverage']!={'completed':2470,'expected':2470}:
        raise ValueError('Full 2470/2470 zero-error initial Top1 >=55% required before navigation')
    if gm['checkpoint_sha256']!=file_sha256(a.checkpoint):raise ValueError('Checkpoint differs from fixed gate evaluation')
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.config import load_config,apply_overrides,validate_config
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    config=apply_overrides(load_config(a.config),['data.root='+str(a.data_root)]);validate_config(config)
    nav,altitude=config['navigation'],config['altitude'];protocol=ClosedLoopProtocol(horizon=nav['horizon'],horizontal_step_m=nav['horizontal_step_m'],vertical_step_m=nav['vertical_step_m'],min_altitude_m=altitude['min_above_ground_m'],max_altitude_m=altitude['max_above_ground_m'],stagnation_steps=nav['stagnation_steps'],success_radius_m=nav['success_distance_m'])
    ds=CityNavDataset(a.data_root,'val_seen',version=config['data']['version']);assert len(ds)==2470
    if file_sha256(ds.annotation_path)!=gm['annotations_sha256']:
        raise ValueError('Navigation episode list differs from full initial evaluation')
    catalog=CityReferCatalog(a.data_root/'cityrefer_original')
    if file_sha256(a.data_root/'cityrefer_original/objects.json')!=gm['objects_sha256']:
        raise ValueError('Navigation public catalog differs from fixed initial evaluation')
    settings=config['state']
    original=PlanningStateBuilder(a.data_root,PublicCatalog(catalog),canvas_size=config['model']['canvas_size'],field_size=config['model']['field_size'],max_height_above_ground_m=settings['max_height_above_ground_m'],trajectory_radius_m=settings['trajectory_radius_m'])
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'rows').mkdir(exist_ok=True)
    builder=NativeObservationBuilder(original,a.data_root/'rgbd',a.output/'observations')
    runtime=V3NavigationPolicy(a.source_v2,a.checkpoint,a.stage1,a.output/'public_inputs',geometry_only=a.geometry_only)
    controller=BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,vertical_step_m=protocol.vertical_step_m,minimum_above_ground_m=protocol.min_altitude_m,maximum_above_ground_m=protocol.max_altitude_m)
    manifest=dict(checkpoint_sha256=file_sha256(a.checkpoint),source_v2_sha256=file_sha256(a.source_v2),stage1_sha256=file_sha256(a.stage1),gate_manifest_sha256=file_sha256(a.initial_eval/'manifest.json'),protocol=asdict(protocol),geometry_only=a.geometry_only,stop_threshold=.3,arrival_guard_m=.001,execution='guarded-blend',n=2470,split='val_seen',annotations_sha256=file_sha256(ds.annotation_path),config_sha256=file_sha256(a.config),code_sha256={x:file_sha256(Path(x)) for x in ['scripts/evaluate_unified_pgvb_v3_navigation.py','graphnav/unified_pgvb_v3_runtime.py','graphnav/a3_runtime.py','graphnav/closedloop.py']})
    mp=a.output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Immutable navigation protocol/checkpoint mismatch')
    atomic_json(manifest,mp)
    for i in range(len(ds)):
        dest=a.output/'rows'/f'{i:05d}.json'
        if dest.exists():continue
        record=ds[i]
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),(),())
        runtime.set_episode(public,builder.transform(record.map_name),catalog)
        # Same original controller/Stop; teacher trajectory/target only enter score_trace AFTER rollout.
        poses,termination,timings,diagnostics=rollout(public,builder,controller,runtime.predict,GROUND_LEVEL[record.map_name],protocol,synchronize=torch.cuda.synchronize)
        metrics=score_trace(poses,record.target_position,record.trajectory,termination,radius=protocol.success_radius_m,decision_seconds=timings)
        atomic_json(dict(index=i,episode_id=list(record.episode_id),poses=poses.tolist(),termination=termination,metrics=metrics,diagnostics=diagnostics),dest)
        if (i+1)%10==0:print(json.dumps(dict(completed=i+1,expected=len(ds))),flush=True)
    rows=[json.loads((a.output/'rows'/f'{i:05d}.json').read_text()) for i in range(len(ds))]
    atomic_json(dict(coverage=dict(completed=len(rows),expected=2470),metrics=aggregate([r['metrics'] for r in rows]),manifest=manifest),a.output/'metrics.json')
    print(json.dumps(json.loads((a.output/'metrics.json').read_text())['metrics'],indent=2))
if __name__=='__main__':main()
