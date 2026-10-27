"""Explicit optional A2 hook on frozen original A0, before original controller."""
import json
import time
import torch
from .paths import bootstrap, CHECKPOINT, A0_SHA, sha
bootstrap()
from graphnav.stage1_runtime import GraphNavStage1Policy
from graphnav.stage1 import preserved_proposal_order
from graphnav.full_localization import GoalField
from .data_protocol import resolve_public_references
from .perception import ObservedBoxPerception
from .observed_geometry import gaussian_belief_mix
from .model import LandmarkBeliefNetwork


def updated_field(original, normalized_grid, logits):
    if torch.equal(logits, original.belief_logits[0].float()):
        return original
    logits=logits[None]
    ids=preserved_proposal_order(logits,original.candidate_ids)
    dense=original.dense_context
    return GoalField(logits,ids,normalized_grid.gather(1,ids[...,None].expand(-1,-1,2)),
                     dense.gather(1,ids[...,None].expand(-1,-1,dense.shape[-1])),
                     dense_context=dense,visual_coverage=original.visual_coverage)


class A2Policy(GraphNavStage1Policy):
    def __init__(self,directory,checkpoint=None,calibration=None,*,collect=True,control='real'):
        if sha(CHECKPOINT)!=A0_SHA:
            raise ValueError('A2 requires archived original A0 checkpoint')
        super().__init__(CHECKPOINT,directory)
        if self.args.grid_size!=28:
            raise ValueError('Original28 grid required')
        self.branch=None
        self.calibration=dict(enabled=False,alpha=0.,threshold=1.,temperature=1.)
        if checkpoint:
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state['a0_sha256']!=A0_SHA:
                raise ValueError('A2 training lineage is not A0')
            self.branch=LandmarkBeliefNetwork(**state['model_config']).to(self.device)
            self.branch.load_state_dict(state['model'],strict=True)
            self.branch.eval().requires_grad_(False)
        if calibration:
            if checkpoint is None:
                raise ValueError('Calibration requires frozen checkpoint')
            self.calibration=json.loads(open(calibration).read())
            if self.calibration['checkpoint_sha256']!=sha(checkpoint):
                raise ValueError('Calibration/checkpoint identity mismatch')
            if not 0<=self.calibration['alpha']<=.5 or self.calibration['temperature']<=0:
                raise ValueError('Invalid calibrated fusion')
        self.collect=collect
        self.control=control
        self.perception=ObservedBoxPerception(self.native) if collect or self.calibration['enabled'] else None
        self.model.register_forward_hook(self._refine)

    def set_episode(self,episode,transform,catalog):
        super().set_episode(episode,transform,catalog)
        # Only named public map reference assets enter A2. Destination identity
        # and coordinates are not passed to resolution or network inputs.
        objects=[o for o in catalog.objects[episode.map_name].values() if o.get('name')]
        self.a2_references=resolve_public_references(episode.map_name,episode.reference_names,objects)

    def _refine(self,module,args,out):
        self.original_field=out.belief
        self.corrected_field=out.belief
        self.pack=self.meta=self.branch_output=None
        self.local_diagnostics=dict(a2_updated=False,a2_fallback=True,a2_alpha=0.,a2_seconds=0.)
        if self.perception is None:
            return out
        tick=time.perf_counter()
        self.pack,self.meta=self.perception.extract(self,self.current_state,out.belief)
        alpha=0.
        if self.branch is not None:
            with torch.autocast('cuda',enabled=False):
                self.branch_output=self.branch(**{k:v[None] for k,v in self.pack['network'].items()},control=self.control)
            result=self.branch_output
            match=float(torch.sigmoid(result['match_logit'][0]/self.calibration['temperature']))
            if self.calibration['enabled'] and match>=self.calibration['threshold'] and self.control!='zero':
                alpha=self.calibration['alpha']
            spatial=self.pack['spatial']
            trusted=result['reliable'][0] & (torch.sigmoid(result['candidate_match_logits'][0]/self.calibration['temperature'])>=self.calibration['threshold'])
            mixed=gaussian_belief_mix(spatial['base_logits'],spatial['grid_world'],
                                     spatial['candidate_xy']+result['offset'][0],result['scores'][0],
                                     spatial['candidate_support'],trusted,alpha)
            out.belief=updated_field(out.belief,self.inputs['grid_xy'],mixed)
            self.corrected_field=out.belief
            self.local_diagnostics['a2_matchability']=match
        changed=out.belief is not self.original_field
        self.local_diagnostics.update(a2_updated=changed,a2_fallback=not changed,a2_alpha=alpha,
                                      a2_candidates=len(self.meta['candidates']),
                                      a2_valid_references=self.meta['valid_references'],
                                      a2_detection_error=self.meta['detector_error'],
                                      a2_seconds=time.perf_counter()-tick)
        return out

    @torch.no_grad()
    def predict(self,state,instruction):
        self.current_state=state
        prediction=super().predict(state,instruction)
        prediction.diagnostics.update(self.local_diagnostics)
        return prediction
