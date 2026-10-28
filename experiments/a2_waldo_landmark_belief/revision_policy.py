"""Opt-in frozen-A0 runtime with the revision model; no private label imports."""
import json,time
from pathlib import Path
import torch
from .paths import ROOT,A0_SHA,sha
from .policy import A2Policy,updated_field
from .revision_model import RevisionNetwork
from .revision_runtime import public_network_inputs,apply_revision


class RevisionPolicy(A2Policy):
    def __init__(self,directory,checkpoint,calibration=None,*,control='real'):
        super().__init__(directory,collect=True,control=control)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['a0_sha256']!=A0_SHA or state['config']['version']!='a2_revision_v1':raise ValueError('Wrong revision lineage')
        for name,digest in state['code_sha256'].items():
            if Path(name).name!=name or sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)!=digest:
                raise ValueError('Frozen revision training implementation changed')
        self.branch=RevisionNetwork(**state['model_config']).to(self.device).eval().requires_grad_(False)
        self.branch.load_state_dict(state['model'],strict=True)
        self.revision_parameters=dict(enabled=False,scale=0.,threshold=1.,temperature=1.)
        if calibration:
            params=json.loads(Path(calibration).read_text())
            if params['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Revision calibration/checkpoint mismatch')
            self.revision_parameters=params

    def _refine(self,module,args,out):
        self.original_field=out.belief;self.corrected_field=out.belief
        start=time.perf_counter()
        self.pack,self.meta=self.perception.extract(self,self.current_state,out.belief)
        network=public_network_inputs(self.pack,self.meta)
        # Internal diagnostic controls do not change model parameters.
        if self.control=='no_text':
            network=dict(network,instruction_tokens=torch.zeros_like(network['instruction_tokens']),
                         landmark_name_tokens=torch.zeros_like(network['landmark_name_tokens']))
        with torch.autocast('cuda',enabled=False):
            self.branch_output=self.branch(**{k:v[None] for k,v in network.items()},control=self.control)
        parameters=self.revision_parameters if self.control!='zero' else dict(self.revision_parameters,enabled=False)
        with torch.autocast('cuda',enabled=False):
            mixed,info=apply_revision(self.pack['spatial'],self.branch_output,parameters)
        out.belief=updated_field(out.belief,self.inputs['grid_xy'],mixed)
        self.corrected_field=out.belief
        self.local_diagnostics=dict(a2_updated=info['updated'],a2_fallback=not info['updated'],a2_alpha=info['gate'],
                                    a2_benefit_probability=info['benefit_probability'],a2_candidates=len(self.meta['candidates']),
                                    a2_valid_references=self.meta['valid_references'],a2_detection_error=self.meta['detector_error'],
                                    a2_seconds=time.perf_counter()-start)
        return out
