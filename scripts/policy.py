"""Label-free optional local refinement before the original controller call."""
from common import ROOT,CHECKPOINT,configuration,sha,load_branch
import json
import time
import numpy as np
import torch
from graphnav.stage1_runtime import GraphNavStage1Policy
from graphnav.stage1 import preserved_proposal_order
from graphnav.full_localization import GoalField
from perception import BoxPerception
from fusion import local_mix


def updated_field(original,grid_xy,logits):
    if torch.equal(logits,original.belief_logits[0].float()):return original
    logits=logits[None];ids=preserved_proposal_order(logits,original.candidate_ids);dense=original.dense_context
    return GoalField(logits,ids,grid_xy.gather(1,ids[...,None].expand(-1,-1,2)),
        dense.gather(1,ids[...,None].expand(-1,-1,dense.shape[-1])),dense_context=dense,visual_coverage=original.visual_coverage)


class RefinementPolicy(GraphNavStage1Policy):
    def __init__(self,directory,checkpoint=None,calibration=None,collect=False,control='real'):
        super().__init__(CHECKPOINT,directory);assert self.args.grid_size==28
        self.branch=None;self.calibration=dict(enabled=False,alpha=0.,threshold=1.,temperature=1.)
        if checkpoint:self.branch,self.branch_state=load_branch(checkpoint,self.device);self.branch.requires_grad_(False)
        if calibration:
            if not checkpoint:raise ValueError('Calibration requires branch checkpoint')
            self.calibration=json.loads(open(calibration).read())
            if self.calibration['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Calibration checkpoint mismatch')
            if self.calibration['config_sha256']!=sha(ROOT/'configs/frozen.json'):raise ValueError('Calibration config mismatch')
        self.collect=collect;self.control=control;self.native.microbatch=8
        self.perception=BoxPerception() if collect or self.calibration['enabled'] else None
        self.pack=None;self.meta=None;self.branch_output=None
        self.model.register_forward_hook(self._refine)

    def _refine(self,module,args,out):
        self.original_field=out.belief;self.corrected_field=out.belief;self.pack=None;self.meta=None;self.branch_output=None
        self.local_diagnostics=dict(local_updated=False,local_candidates=0,local_seconds=0.,local_alpha=0.)
        if not self.collect and (not self.calibration['enabled'] or self.control=='zero'):return out
        tick=time.perf_counter();self.pack,self.meta=self.perception.extract(self,self.current_state,out.belief)
        if self.branch is not None and self.calibration['enabled'] and self.control!='zero':
            with torch.autocast('cuda',enabled=False):self.branch_output=self.branch(**self.pack['network'],control=self.control)
            result=self.branch_output;s=self.pack['spatial'];cal=self.calibration
            match=float(torch.sigmoid(result['match_logit']/cal['temperature']))
            alpha=cal['alpha'] if match>=cal['threshold'] else 0.
            # Bounded offset is clipped to box+2m by construction; unsupported
            # extrapolation cannot introduce new grid cells beyond cached support.
            mixed=local_mix(s['base_logits'],s['grid_world'],s['candidate_xy']+result['offset'],result['scores'],s['candidate_support'],alpha)
            out.belief=updated_field(out.belief,self.inputs['grid_xy'],mixed['logits']);self.corrected_field=out.belief
            self.local_diagnostics.update(local_updated=out.belief is not self.original_field,local_alpha=alpha,local_matchability=match)
        self.local_diagnostics.update(local_candidates=len(self.meta['candidates']),local_seconds=time.perf_counter()-tick)
        return out

    @torch.no_grad()
    def predict(self,state,instruction):
        self.current_state=state;result=super().predict(state,instruction);result.diagnostics.update(self.local_diagnostics)
        if hasattr(self,'corrected_field'):
            i=int(result.diagnostics['candidate']);cell=int(self.corrected_field.candidate_ids[0,i]);expected=self.world[0,cell].float().cpu().numpy()
            assert np.allclose(expected,result.diagnostics['selected_goal_xy'],atol=1e-3)
        return result
