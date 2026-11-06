"""Two TRAIN episodes for interface acceptance, with no target/scoring access."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from graphnav.checkpointing import atomic_json
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout
from graphnav.unified_angle_only_runtime import UnifiedAngleOnlyPolicy


def main():
 p=argparse.ArgumentParser()
 for k in ('stage1','original-v2','checkpoint','angle-checkpoint','public-objects','data-root','sbf-source','config','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.sbf_source));torch.set_num_threads(4)
 from sbfnav.dataset import CityNavDataset,CityReferCatalog
 from sbfnav.config import load_config
 from sbfnav.planning_state import PlanningStateBuilder
 from sbfnav.navigation import BoundedWaypointController
 from sbfnav.mapdata import GROUND_LEVEL
 from graphnav.dynamic_runtime import NativeObservationBuilder
 import graphnav.unified_angle_only_belief as angular
 def forbidden_labels(*args,**kwargs):raise AssertionError('Oracle supervision called during runtime')
 angular.bearing_supervision=forbidden_labels
 cfg=load_config(a.config);data=CityNavDataset(a.data_root,'train_seen',version=cfg['data']['version']);cat=CityReferCatalog(a.data_root/'cityrefer')
 builder=NativeObservationBuilder(PlanningStateBuilder(a.data_root,PublicCatalog(cat),canvas_size=cfg['model']['canvas_size'],field_size=cfg['model']['field_size'],max_height_above_ground_m=cfg['state']['max_height_above_ground_m'],trajectory_radius_m=cfg['state']['trajectory_radius_m']),a.data_root/'rgbd',a.output/'observations')
 proto=ClosedLoopProtocol();ctrl=BoundedWaypointController(horizontal_step_m=proto.horizontal_step_m,vertical_step_m=proto.vertical_step_m,minimum_above_ground_m=proto.min_altitude_m,maximum_above_ground_m=proto.max_altitude_m)
 model=UnifiedAngleOnlyPolicy(a.stage1,a.original_v2,a.checkpoint,a.output/'public_inputs',angle_head_checkpoint=a.angle_checkpoint,public_objects=a.public_objects);results=[]
 for i in (0,1):
  r=data[i];ep=PublicEpisode(r.map_name,r.instruction,r.start_pose.copy(),(),())
  # This facade cannot resolve private target IDs or read processed annotations.
  model.set_episode(ep,builder.transform(ep.map_name),SimpleNamespace(objects=cat.objects))
  poses,end,times,diag=rollout(ep,builder,ctrl,model.predict,GROUND_LEVEL[ep.map_name],proto,synchronize=torch.cuda.synchronize)
  assert diag and np.isfinite(poses).all() and model.record['target_xy']==[0.,0.]
  assert all(d['modality']=='predicted_angular_direction_only' and d['angle_GT_used_at_inference'] is False for d in diag)
  assert all(np.isfinite(d['angle_delta_abs_max']) and d['angle_delta_abs_max']<=1 for d in diag)
  results.append(dict(TRAIN_index=i,steps=len(poses)-1,termination=end,diagnostics=diag,poses=poses.tolist(),GT_used=False))
 atomic_json(dict(interface_passed=True,performance_claim=False,rows=results),a.output/'smoke.json')
if __name__=='__main__':main()
