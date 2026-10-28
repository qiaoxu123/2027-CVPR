"""Frozen original A0 + completed relation model + trained direct-risk head."""
import json,time
from pathlib import Path
import torch
from .paths import ROOT,A0_SHA,sha
from .revision_policy import RevisionPolicy
from .revision_runtime import public_network_inputs
from .policy import updated_field
from .direct_risk_head import DirectRiskHead
from .direct_risk_runtime import risk_forward


class DirectRiskPolicy(RevisionPolicy):
    def __init__(self,directory,rank_checkpoint,head_checkpoint,calibration=None,*,control='real'):
        # The inherited original model hook binds this class's _refine. No
        # inference happens before both frozen added components are loaded.
        super().__init__(directory,rank_checkpoint,calibration=None,control=control)
        head_checkpoint=Path(head_checkpoint);training=head_checkpoint.parent
        state=torch.load(head_checkpoint,map_location='cpu',weights_only=False)
        protocol=json.loads((training/'protocol.json').read_text());marker=json.loads((training/'training_complete.json').read_text())
        proof=json.loads((training/'artifact_verification.json').read_text())
        if (protocol['config']['version']!='a2_direct_risk_v2' or protocol['a0_sha256']!=A0_SHA
            or protocol['lineage']['rank_checkpoint_sha256']!=sha(rank_checkpoint) or state['rank_checkpoint_sha256']!=sha(rank_checkpoint)
            or state['signature_sha256']!=sha(training/'protocol.json') or marker['signature_sha256']!=state['signature_sha256']
            or marker['status']!='direct_risk_full25_complete_confirmation_pending' or marker['additional_discrimination_epochs']!=20 or marker['additional_refinement_epochs']!=5
            or marker['checkpoint_sha256']!=sha(head_checkpoint) or marker['selected_epoch']!=state['epoch'] or not marker['checkpoint_eligible']
            or proof['status']!='direct_risk_all25_artifacts_verified' or proof['checkpoint_sha256']!=sha(head_checkpoint) or not proof['all_dev_epochs_recomputed']):
            raise ValueError('Completed, eligible, provenance-bound direct-risk head required')
        for name,digest in protocol['code_sha256'].items():
            if Path(name).name!=name or sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)!=digest:raise ValueError('Frozen direct-risk training source changed')
        weights=state['model'];self.risk_head=DirectRiskHead(weights['mean'],weights['std'],weights['class_weights']).to(self.device).eval().requires_grad_(False)
        self.risk_head.load_state_dict(weights,strict=True)
        self.risk_parameters=dict(enabled=False,minimum_gain=1.,maximum_harm=0.,temperature=1.)
        if calibration:
            parameters=json.loads(Path(calibration).read_text())
            if parameters['checkpoint_sha256']!=sha(head_checkpoint) or parameters['rank_checkpoint_sha256']!=sha(rank_checkpoint):raise ValueError('Risk calibration bound to different weights')
            self.risk_parameters=parameters

    def _refine(self,module,args,out):
        self.original_field=out.belief;self.corrected_field=out.belief;start=time.perf_counter()
        self.pack,self.meta=self.perception.extract(self,self.current_state,out.belief)
        network=public_network_inputs(self.pack,self.meta)
        if self.control=='no_text':
            network=dict(network,instruction_tokens=torch.zeros_like(network['instruction_tokens']),landmark_name_tokens=torch.zeros_like(network['landmark_name_tokens']))
        parameters=self.risk_parameters if self.control!='zero' else dict(self.risk_parameters,enabled=False)
        with torch.autocast('cuda',enabled=False):
            self.branch_output=self.branch(**{k:v[None] for k,v in network.items()},control=self.control)
            mixed,info,self.risk_output=risk_forward(self.pack['spatial'],self.branch_output,self.risk_head,parameters)
        out.belief=updated_field(out.belief,self.inputs['grid_xy'],mixed);self.corrected_field=out.belief
        self.local_diagnostics=dict(a2_updated=info['updated'],a2_fallback=not info['updated'],a2_alpha=info['gate'],
            a2_gain_score=info['gain_score'],a2_harm_score=info['harm_score'],a2_candidates=len(self.meta['candidates']),
            a2_valid_references=self.meta['valid_references'],a2_detection_error=self.meta['detector_error'],a2_seconds=time.perf_counter()-start)
        return out
