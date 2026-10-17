"""Explicit frozen-controller interventions, separate from deployed runtime."""
from dataclasses import replace
import numpy as np
import torch
from graphnav.a3_runtime import A3TrajectoryPolicy

ORIGINAL_PREDICT_FIELD=A3TrajectoryPolicy.predict_field
ARMS=('predicted_direct','predicted_waypoint','predicted_guarded_blend','oracle_direct','oracle_guarded_blend','rule_arrival')

def controlled_predict_field(self,state,field):
    arm=getattr(self,'control_arm',None)
    if arm is None:return ORIGINAL_PREDICT_FIELD(self,state,field)
    if arm not in ARMS:raise ValueError(arm)
    original_xy=field.candidate_xy.clone()
    if arm.startswith('oracle_'):
        # ONLY explicit diagnostic controller access to ground truth. Never
        # passed to language/visual field construction or checkpoint selection.
        bounds=torch.as_tensor(self.record['world_bounds'],device=self.device)
        xy=field.candidate_xy.clone()
        xy[:,0]=(torch.as_tensor(self.oracle_goal_xy,device=self.device)-bounds[:2])/(bounds[2:]-bounds[:2])
        field=replace(field,candidate_xy=xy)
    execution='guarded-direct' if arm.endswith('direct') else 'fixed-goal' if arm=='predicted_waypoint' else 'guarded-blend'
    saved=self.execution;self.execution=execution
    try:result=ORIGINAL_PREDICT_FIELD(self,state,field)
    finally:self.execution=saved
    goal=np.asarray(result.diagnostics['selected_goal_xy'])
    distance=float(np.linalg.norm(np.asarray(state.pose)[:2]-goal))
    arrival=distance<=.001;altitude=bool(result.diagnostics['stop_altitude_guard'])
    learned=result.diagnostics['stop_probability']>=.3
    result.should_stop=arrival and altitude and (True if arm=='rule_arrival' else learned)
    result.diagnostics.update(control_arm=arm,oracle_diagnostic=arm.startswith('oracle_'),
        predicted_goal_distance=distance,predicted_arrival=arrival,learned_stop_requested=learned,
        policy_stop_requested=result.should_stop,arrival_guard_m=.001,success_radius_m=20.,
        unmodified_candidate0_xy=original_xy[0,0].float().cpu().tolist())
    return result
