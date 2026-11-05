"""Two genuine TRAIN closed-loop interface checks, never a performance subset."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout
from graphnav.checkpointing import atomic_json


def main():
    p=argparse.ArgumentParser()
    for k in ('stage1','original-v2','checkpoint','data-root','sbf-source','config','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.config import load_config
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.unified_full_relation_runtime import UnifiedPGVBFullRelationPolicy
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    cfg=load_config(a.config);data=CityNavDataset(a.data_root,'train_seen',version=cfg['data']['version']);cat=CityReferCatalog(a.data_root/'cityrefer')
    builder=NativeObservationBuilder(PlanningStateBuilder(a.data_root,PublicCatalog(cat),canvas_size=cfg['model']['canvas_size'],field_size=cfg['model']['field_size'],max_height_above_ground_m=cfg['state']['max_height_above_ground_m'],trajectory_radius_m=cfg['state']['trajectory_radius_m']),a.data_root/'rgbd',a.output/'observations')
    protocol=ClosedLoopProtocol();controller=BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,vertical_step_m=protocol.vertical_step_m,minimum_above_ground_m=protocol.min_altitude_m,maximum_above_ground_m=protocol.max_altitude_m)
    runtime=UnifiedPGVBFullRelationPolicy(a.stage1,a.original_v2,a.checkpoint,a.output/'inputs')
    report=[]
    for index in (0,1):
        row=data[index];ep=PublicEpisode(row.map_name,row.instruction,row.start_pose.copy(),tuple(cat.referenced_landmarks(row)),tuple(cat.referenced_landmark_names(row)))
        runtime.set_episode(ep,builder.transform(ep.map_name),cat)
        poses,termination,timings,diagnostics=rollout(ep,builder,controller,runtime.predict,GROUND_LEVEL[ep.map_name],protocol,synchronize=torch.cuda.synchronize)
        assert len(diagnostics)>0 and np.isfinite(poses).all()
        assert all(np.isfinite(d['relation_delta_max']) for d in diagnostics)
        report.append(dict(index=index,TRAIN=True,termination=termination,steps=len(poses)-1,relation_delta_max=max(d['relation_delta_max'] for d in diagnostics),ms=float(np.mean(timings)*1000),RGB_active=sum(d['rgb_active'] for d in diagnostics),poses=poses.tolist()))
    atomic_json(dict(interface_passed=True,episodes=report,performance_claim=False,GT_scored=False),a.output/'smoke.json')

if __name__=='__main__':main()
