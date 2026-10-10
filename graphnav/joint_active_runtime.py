"""Joint checkpoint runtime: current/history observations, bounded active actions."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .a3_runtime import public_static_record
from .active_observation import ActiveObservationConfig,plan_active_observation
from .appearance_matching import ObservedSigLIPMatchingEncoder
from .candidate_geometry import FullCandidateDataset,collate_full
from .full_experiment import _batch_inputs,new_model
from .joint_active import JointActiveGraphNav
from .observed_patch_scales import ObservedPatchScaleEncoder
from .token_language import FrozenBERTTokens
from .trajectory_policy import calibrated_stop_probability
from .joint_inference_controls import joint_stop_decision


class JointActivePolicy:
    def __init__(self,checkpoint,directory,*,device='cuda',active_observation=False,
                 stop_guard_m=20.0,stop_threshold=None,visual_control='full'):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        protocol=state['protocol']
        if protocol['name']!='joint-active-v1' or not protocol['trainable_a3']:
            raise ValueError('complete joint checkpoint required')
        self.args=SimpleNamespace(**state['a3_args']);self.protocol=protocol
        self.active_observation=bool(active_observation)
        self.stop_guard_m=float(stop_guard_m)
        self.stop_threshold=float(protocol.get('stop_threshold', .5) if stop_threshold is None else stop_threshold)
        if visual_control not in ('full','no-belief','zero-vectors','shuffle-vectors'):
            raise ValueError('invalid visual control mode')
        self.visual_control=visual_control
        self.device=torch.device(device);self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.model=JointActiveGraphNav(new_model(self.args,self.device)).to(self.device).eval()
        self.model.load_state_dict(state['model'],strict=True)
        self.bert=FrozenBERTTokens(name=self.args.text_backbone,max_length=getattr(self.args,'max_text_length',96),device=device)
        self.native=ObservedSigLIPMatchingEncoder(protocol['backbone'],device=device,microbatch=1)
        if getattr(self.native.model.config,'_commit_hash',None)!=protocol['backbone_revision']:
            raise ValueError('frozen visual backbone revision mismatch')
        self.visual=ObservedPatchScaleEncoder(self.native)
        self.active_config=ActiveObservationConfig(max_actions=protocol['active_max_actions'])

    def set_episode(self,episode,transform,catalog):
        self.record=public_static_record(episode,transform,catalog,max_landmarks=self.args.max_landmarks)
        self.active_actions=0;self.inputs=None;self.previous_pose=None
        self.feature_sum=self.feature_count=self.quality_sum=None

    @torch.no_grad()
    def predict(self,state,instruction):
        from sbfnav.mapdata import GROUND_LEVEL
        if instruction!=self.record['instruction']:raise ValueError('episode language changed')
        if self.inputs is None:
            path=self.directory/'current_public_query.jsonl';path.write_text(json.dumps(self.record)+'\n')
            ds=FullCandidateDataset(path,grid_size=self.args.grid_size,max_landmarks=self.args.max_landmarks,
                max_roads=self.args.max_roads,max_relations=self.args.max_relations,
                prior_sigma_m=self.args.prior_sigma_m,rule_sigma_m=self.args.rule_sigma_m)
            static=collate_full([ds[0]])
            self.inputs,self.tokens,self.mask,self.names=_batch_inputs(static,self.bert,self.device,'geometry')
            self.native_text=self.native.texts([instruction]).detach().clone()
        bounds=torch.tensor(self.record['world_bounds'],device=self.device,dtype=torch.float32)[None]
        grid=self.inputs['grid_xy'];world=bounds[:,None,:2]+grid*(bounds[:,None,2:]-bounds[:,None,:2])
        features,valid,quality=self.visual.encode([state.current_observation],[world[0].cpu().numpy()])[0]
        features=features.float().to(self.device);valid=valid.to(self.device);quality=quality.to(self.device)
        # Preserve source coverage, candidate positions and quality in ablations.
        if self.visual_control == 'zero-vectors':
            features = torch.zeros_like(features)
        elif self.visual_control == 'shuffle-vectors':
            present = valid.any(-1).nonzero().flatten()
            if len(present) > 1:
                features = features.clone()
                features[present] = features[present].roll(1, dims=0)
        if self.feature_sum is None:
            self.feature_sum=torch.zeros_like(features);self.feature_count=torch.zeros_like(valid,dtype=torch.float32)
            self.quality_sum=torch.zeros_like(quality)
        # No accumulation of repeated stationary views or yaw-only pseudo-detail.
        new_pose=self.previous_pose is None or np.linalg.norm(np.asarray(state.pose)[:3]-self.previous_pose[:3])>=2.
        if new_pose:
            self.feature_sum+=features*valid[...,None];self.feature_count+=valid
            self.quality_sum+=quality*valid[...,None];self.previous_pose=np.asarray(state.pose).copy()
        images=self.feature_sum/self.feature_count[...,None].clamp_min(1)
        observed=self.feature_count>0
        averaged_quality=self.quality_sum/self.feature_count[...,None].clamp_min(1)
        ground=GROUND_LEVEL[self.record['map_name']]
        pose=torch.tensor(np.asarray(state.pose),device=self.device,dtype=torch.float32)[None]
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            output=self.model(static_inputs=self.inputs,tokens=self.tokens,token_mask=self.mask,node_emb=self.names,
                native_text=self.native_text,roi_features=images[None],roi_valid=observed[None],
                roi_quality=averaged_quality[None],pose_world=pose,bounds=bounds,
                ground_level=torch.tensor([ground],device=self.device),
                visual_control='no-belief' if self.visual_control=='no-belief' else 'full')
        candidate,mode,waypoint,stop=output.trajectory.select()
        stop_probability=float(calibrated_stop_probability(stop,self.protocol['stop_positive_weight'])[0])
        selected_goal = bounds[0,:2] + output.belief.candidate_xy[0,candidate[0]] * (bounds[0,2:]-bounds[0,:2])
        should_stop, stop_diag = joint_stop_decision(
            pose_world=state.pose, candidate_world_xy=selected_goal.float().cpu().numpy(),
            stop_probability=stop_probability, ground_level=ground,
            threshold=self.stop_threshold, guard_radius_m=self.stop_guard_m,
        )
        diagnostic=dict(policy='Joint trainable A3 + observed multiscale/history + trained trajectory/Stop',
            stop_probability=stop_probability,observed_cells=int(output.observed.sum()),
            visual_gain=float(self.model.visual_gain.tanh()),active_actions_used=self.active_actions,
            active_observation_enabled=self.active_observation,visual_control=self.visual_control,
            selected_predicted_goal_xy=selected_goal.float().cpu().tolist(),**stop_diag,
            trajectory_candidate=int(candidate[0]),trajectory_mode=int(mode[0]))
        top=output.belief.belief_logits.float().topk(8,-1).indices
        diagnostic['belief_top8_world_xy']=world.gather(1,top[...,None].expand(-1,-1,2))[0].cpu().tolist()
        diagnostic['observed_grid_ids']=output.observed[0].nonzero().flatten().cpu().tolist()
        nearest_appearance=output.appearance_scores.masked_fill(~output.observed,-1e4).argmax(-1)
        diagnostic['appearance_top1_world_xy']=(world[0,nearest_appearance[0]].cpu().tolist()
            if output.observed.any() else None)
        confidence=output.appearance_scores.masked_fill(~output.observed,-1e4).softmax(-1)*output.observed
        if self.active_observation and not should_stop:
            decision=plan_active_observation(pose_xyzyaw=state.pose,map_bounds=self.record['world_bounds'],
                ground_level=ground,candidate_xy=grid[0].cpu().numpy(),
                candidate_logits=output.belief.belief_logits[0].float().cpu().numpy(),
                observed_confidence=confidence[0].cpu().numpy(),instruction=instruction,step=state.step,
                used_actions=self.active_actions,config=self.active_config)
            if decision is not None:
                self.active_actions+=1
                diagnostic.update(active_action=decision.action,active_reason=decision.reason,
                    active_actions_used=self.active_actions,estimated_object_pixels=decision.estimated_object_pixels)
                return SimpleNamespace(xyz=decision.xyz,should_stop=False,diagnostics=diagnostic)
        return SimpleNamespace(xyz=waypoint[0].float().cpu().numpy(),should_stop=should_stop,diagnostics=diagnostic)
