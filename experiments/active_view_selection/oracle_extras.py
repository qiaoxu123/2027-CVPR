"""Privileged offline GT-directed diagnostic. Never imported by a policy."""
import json,time,math
from pathlib import Path
import numpy as np,torch
from .run import R,Simulator,dump,current_state,summarize_trace,digest
from .planner import boxes,legal,View
from experiments.multiview_rgb_grounding.render import OBJECTS

def run():
 assert not (R/'oracle_extra_predictions.json').exists();start=time.time();states=torch.load(R/'cohort.pt',weights_only=True);objects=json.loads(OBJECTS.read_text());gate=json.loads(Path('experiments/multiview_rgb_grounding/protocol.json').read_text())['quality_gates'];normal=json.loads((R/'predictions.json').read_text());rows=[];sim=None;name=None
 for row in states:
  if time.time()-start>600:break
  m=row['metadata'];mapname=m['map_name'];bb=boxes(objects[mapname]);target=row['label']['target_object_id'];goal=np.asarray(objects[mapname][target]['contour']).mean(0);initial=current_state(row);pose=np.asarray(initial['xyz']);poses=[pose.copy()]
  for step in range(2):
   d=goal-pose[:2];dist=np.linalg.norm(d);nxt=pose.copy();nxt[:2]+=d*min(1,50/max(dist,1e-8));nxt[2]+=float(np.clip(row['ground']+50-pose[2],-10,10))
   if not legal(pose,nxt,row['public']['bounds'],row['ground'],bb):break
   poses.append(nxt);pose=nxt
  # Simulator supports prior-frame metadata loading to avoid rerendering.
  if mapname!=name:sim=Simulator(mapname,objects[mapname],gate);name=mapname
  base=View(tuple(initial['xyz']),initial['uav_yaw'],0,'initial',initial['uav_yaw']);baseline=sim.exposure(base);by_step=[]
  for i,pose in enumerate(poses):
   d=goal-pose[:2];yaw=math.atan2(d[1],d[0]) if np.linalg.norm(d)>1e-8 else initial['uav_yaw'];vs=[]
   for tilt in [0,20,40,60]:
    v=View(tuple(pose),yaw if tilt else initial['uav_yaw'],tilt,'offline_GT_approach' if i else 'offline_GT_camera',initial['uav_yaw']);meta,win=sim.exposure(v);vs.append((v,meta,win))
   by_step.append(vs)
  traces=[]
  for v in by_step[0]:traces.append([v,v])
  if len(by_step)>1:
   for a in by_step[0]+by_step[1]:
    for b in by_step[1]:
     if legal(a[0].xyz,b[0].xyz,row['public']['bounds'],row['ground'],bb):traces.append([a,b])
  if len(by_step)>2:
   for a in by_step[1]:
    for b in by_step[2]:traces.append([a,b])
  candidates=[summarize_trace(row,'target_directed_oracle',t,baseline,sim) for t in traces];old=next(x for x in normal if x['sample_id']==m['sample_id'] and x['strategy']=='view_oracle');candidates.append(dict(old,strategy='target_directed_oracle'));chosen=max(candidates,key=lambda x:(x['metrics']['target_credible'],x['metrics']['max_native_target_side'],-x['metrics']['xy_m']-x['metrics']['z_m']));rows.append(chosen);dump(R/'oracle_extra_partial.json',rows);print(json.dumps(dict(families=len(rows),total=len(states),oracle_credible=chosen['metrics']['target_credible'],seconds=time.time()-start)),flush=True)
 dump(R/'oracle_extra_predictions.json',rows);dump(R/'oracle_extra_execution.json',dict(seconds=time.time()-start,families=len(rows),training=False,GT_policy=True,normal_policy_affected=False,protocol_sha256=digest(Path(__file__).with_name('oracle_extras_protocol.json'))))

if __name__=='__main__':run()
