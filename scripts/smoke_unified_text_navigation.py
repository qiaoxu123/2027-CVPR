"""TRAIN-only real navigation interface smoke, never SR/model selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from graphnav.checkpointing import atomic_json
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout
from graphnav.unified_text_ablation_runtime import UnifiedTextAblationPolicy


def main():
    p=argparse.ArgumentParser()
    for k in ('stage1','original-v2','checkpoint','public-objects','data-root','sbf-source','config','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.sbf_source));torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.config import load_config
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.dynamic_runtime import NativeObservationBuilder
    cfg=load_config(a.config);data=CityNavDataset(a.data_root,'train_seen',version=cfg['data']['version']);cat=CityReferCatalog(a.data_root/'cityrefer')
    builder=NativeObservationBuilder(PlanningStateBuilder(a.data_root,PublicCatalog(cat),canvas_size=cfg['model']['canvas_size'],field_size=cfg['model']['field_size'],max_height_above_ground_m=cfg['state']['max_height_above_ground_m'],trajectory_radius_m=cfg['state']['trajectory_radius_m']),a.data_root/'rgbd',a.output/'observations')
    protocol=ClosedLoopProtocol();controller=BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,vertical_step_m=protocol.vertical_step_m,minimum_above_ground_m=protocol.min_altitude_m,maximum_above_ground_m=protocol.max_altitude_m);rows=[]
    for mode in ('full_text','landmark_relation'):
        policy=UnifiedTextAblationPolicy(a.stage1,a.original_v2,a.checkpoint,a.output/mode,instruction_mode=mode,public_objects=a.public_objects)
        for index in (0,1):
            r=data[index];ep=PublicEpisode(r.map_name,r.instruction,r.start_pose.copy(),(),());policy.set_episode(ep,builder.transform(ep.map_name),cat)
            poses,end,times,diag=rollout(ep,builder,controller,policy.predict,GROUND_LEVEL[ep.map_name],protocol,synchronize=torch.cuda.synchronize)
            assert diag and np.isfinite(poses).all();assert policy.record['target_xy']==[0.,0.]
            assert all(d['rgb_active'] is False and np.isfinite(d['relation_delta_max']) for d in diag)
            rows.append(dict(mode=mode,TRAIN_index=index,model_instruction=policy.record['instruction'],record=policy.record,termination=end,steps=len(poses)-1,mean_ms=float(np.mean(times)*1000),diagnostics=diag,poses=poses.tolist(),GT_used=False))
        del policy;torch.cuda.empty_cache()
    atomic_json(dict(interface_passed=True,performance_claim=False,checkpoint=str(a.checkpoint),rows=rows),a.output/'smoke.json')

if __name__=='__main__':main()
