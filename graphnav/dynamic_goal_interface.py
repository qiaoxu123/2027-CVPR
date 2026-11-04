"""Opt-in causal controller interfaces; no labels, Polygon distances or new weights."""
from dataclasses import replace
import math
import torch
from .stage1_runtime import GraphNavStage1Policy
from .pgvb_navigation_runtime import PGVBNamedLandmarkPolicy,bounded_pgvb_field


def grid_arrival_tolerance(bounds,grid_size):
    x0,y0,x1,y1=map(float,bounds)
    if grid_size<=0 or x1<=x0 or y1<=y0:raise ValueError('Valid public world grid required')
    # Half-cell diagonal is the location quantization uncertainty, not GT error.
    return min(20.,math.hypot((x1-x0)/grid_size,(y1-y0)/grid_size)/2)


def interface_stop(*,probability,distance,altitude_guard,pending,tolerance):
    return bool(probability>=.3 and distance<=tolerance and altitude_guard and not pending)


class CandidateCommitment:
    """Defer a challenger by at most one decision; never hold an evicted candidate.

    Two consecutive same challengers confirm immediately on decision two. If
    challengers alternate, the second is accepted anyway at the two-decision
    deadline. This prevents permanently locking an initially wrong hypothesis.
    """
    def __init__(self):self.incumbent=None;self.age=0;self.challenger=None

    def update(self,field,grid_uv):
        if len(field.candidate_ids)!=1:raise ValueError('Single causal episode only')
        ids=field.candidate_ids[0];raw=int(ids[0]);pending=False;reason='unchanged'
        if self.incumbent is None:reason='initial';self.incumbent=raw
        elif raw==self.incumbent:self.age=0;self.challenger=None
        elif not bool((ids==self.incumbent).any()):reason='incumbent_evicted';self.incumbent=raw;self.age=0;self.challenger=None
        else:
            self.age+=1
            if self.challenger==raw or self.age>=2:
                reason='confirmed' if self.challenger==raw else 'bounded_deadline';self.incumbent=raw;self.age=0;self.challenger=None
            else:pending=True;reason='confirm_challenger';self.challenger=raw
        audit=dict(pending=pending,reason=reason,raw_candidate_id=raw,controller_candidate_id=self.incumbent,deferral_age=self.age)
        if raw==self.incumbent:return field,audit
        if field.dense_context is None:raise ValueError('Pretrained dense context required')
        ordered=torch.cat((ids[ids==self.incumbent],ids[ids!=self.incumbent]))[None]
        # Preserve every belief value. Only the candidate order given to the
        # frozen head changes; aligned world coordinates/context are re-gathered.
        updated=replace(field,candidate_ids=ordered,candidate_xy=grid_uv.gather(1,ordered[...,None].expand(-1,-1,2)),candidate_context=field.dense_context.gather(1,ordered[...,None].expand(-1,-1,field.dense_context.shape[-1])))
        return updated,audit


class DynamicGoalPGVBPolicy(PGVBNamedLandmarkPolicy):
    """Same frozen E4 evidence, optional candidate commitment and arrival interface."""
    def __init__(self,*args,stabilize=False,arrival=False,**kwargs):
        super().__init__(*args,**kwargs);self.stabilize=bool(stabilize);self.arrival=bool(arrival)

    def set_episode(self,*args,**kwargs):
        super().set_episode(*args,**kwargs);self.commitment=CandidateCommitment()
        self.arrival_tolerance=grid_arrival_tolerance(self.record['world_bounds'],self.args.grid_size)

    @torch.no_grad()
    def predict(self,state,instruction):
        evidence,reason=self._timed_scores(state.current_observation)
        result=dict(pgvb_enabled=True,pgvb_reason=reason,pgvb_active=False,pgvb_covered_cells=0,pgvb_delta_max_abs=0.,pgvb_changed_top1=False)
        selection=dict(pending=False)
        def transform(module,inputs,output):
            original=output.belief;field=original
            if evidence is not None:
                uv,scores,valid=evidence;field,audit=bounded_pgvb_field(field,self.inputs['grid_xy'],uv,scores,valid,bound=self.bound)
                result.update(pgvb_active=audit['active'],pgvb_covered_cells=audit['covered_cells'],pgvb_delta_max_abs=audit['max_abs_delta'],pgvb_changed_top1=audit.get('changed_top1',False))
            raw=int(field.candidate_ids[0,0])
            if self.stabilize:
                field,a=self.commitment.update(field,self.inputs['grid_xy']);selection.update(a)
            else:selection.update(raw_candidate_id=raw,controller_candidate_id=raw,reason='disabled')
            if self.audit_callback is not None:self.audit_callback(original,field,evidence)
            return replace(output,belief=field)
        hook=self.model.register_forward_hook(transform)
        try:prediction=GraphNavStage1Policy.predict(self,state,instruction)
        finally:hook.remove()
        original_stop=bool(prediction.should_stop)
        tolerance=self.arrival_tolerance if self.arrival else 1e-3
        prediction.should_stop=interface_stop(probability=prediction.diagnostics['stop_probability'],distance=prediction.diagnostics['predicted_goal_distance'],altitude_guard=prediction.diagnostics['stop_altitude_guard'],pending=selection['pending'],tolerance=tolerance)
        self.pgvb_calls+=int(result['pgvb_active'])
        prediction.diagnostics.update(result,pgvb_seconds=self.pgvb_seconds,stability_enabled=self.stabilize,arrival_interface_enabled=self.arrival,arrival_tolerance_m=tolerance,original_mm_guard_should_stop=original_stop,goal_commitment=selection,actual_should_stop=prediction.should_stop)
        return prediction
