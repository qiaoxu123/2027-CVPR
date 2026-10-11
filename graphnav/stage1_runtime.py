"""Complete Stage1 checkpoint runtime; original Guarded-Blend control, no active sensing."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .a3_runtime import A3TrajectoryPolicy,public_static_record
from .appearance_matching import ObservedSigLIPMatchingEncoder
from .candidate_geometry import FullCandidateDataset,collate_full
from .evidence_grounding import appearance_query
from .full_experiment import _batch_inputs,new_model
from .observed_patch_scales import ObservedPatchScaleEncoder
from .stage1 import GraphNavStage1
from .stage1_data import CandidateMemory
from .token_language import FrozenBERTTokens


def retrieval_ids(logits):
    """Preserve legacy topk(8) tie order, extend it with eight additional best cells."""
    scores=logits.float();primary=scores.topk(min(8,scores.shape[-1]),-1).indices
    extra=scores.topk(min(16,scores.shape[-1]),-1).indices;rows=[]
    for a,b in zip(primary,extra):
        rest=b[~(b[:,None]==a[None]).any(-1)]
        rows.append(torch.cat((a,rest))[:min(16,scores.shape[-1])])
    return torch.stack(rows)


class GraphNavStage1Policy:
    def __init__(self,checkpoint,directory,*,device='cuda',visual_control='real',history_control='checkpoint'):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.protocol=state['protocol']
        if self.protocol['name']!='graphnav-v1-stage1' or self.protocol['active_observation']:raise ValueError('Stage1 complete non-active checkpoint required')
        self.args=SimpleNamespace(**state['a3_args']);self.device=torch.device(device)
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.model=GraphNavStage1(new_model(self.args,self.device),residual_bound=self.protocol['residual_bound']).to(self.device)
        self.model.load_state_dict(state['model'],strict=True);self.model.requires_grad_(False);self.model.eval()
        self.bert=FrozenBERTTokens(self.args.text_backbone,device=device,max_length=getattr(self.args,'max_text_length',96))
        expected_bert=state.get('config',{}).get('frozen_backbones',{}).get('bert_revision')
        if expected_bert and self.bert.revision!=expected_bert:raise ValueError('frozen BERT revision mismatch')
        self.native=ObservedSigLIPMatchingEncoder(self.protocol['backbone'],device=device,microbatch=1)
        if getattr(self.native.model.config,'_commit_hash',None)!=self.protocol['backbone_revision']:raise ValueError('frozen native backbone revision mismatch')
        self.visual=ObservedPatchScaleEncoder(self.native);self.visual_control=visual_control
        self.use_history=self.protocol['use_history'] if history_control=='checkpoint' else history_control=='history'
        self.head=self.model.trajectory;self.stop_threshold=.3;self.stop_positive_weight=20.;self.execution='guarded-blend';self.explore_candidates=0

    def set_episode(self,episode,transform,catalog):
        self.record=public_static_record(episode,transform,catalog,max_landmarks=self.args.max_landmarks)
        self.visited_goal_cells=set();self.inputs=None;self.memory=None

    @torch.no_grad()
    def predict(self,state,instruction):
        from sbfnav.mapdata import GROUND_LEVEL
        if instruction!=self.record['instruction']:raise ValueError('language changed within episode')
        if self.inputs is None:
            path=self.directory/'current_public_query.jsonl';path.write_text(json.dumps(self.record)+'\n')
            ds=FullCandidateDataset(path,grid_size=self.args.grid_size,max_landmarks=self.args.max_landmarks,max_roads=self.args.max_roads,max_relations=self.args.max_relations,prior_sigma_m=self.args.prior_sigma_m,rule_sigma_m=self.args.rule_sigma_m)
            self.inputs,self.tokens,self.mask,self.names=_batch_inputs(collate_full([ds[0]]),self.bert,self.device,'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16):self.base=self.model.a3(self.inputs,variant='geometry',tokens=self.tokens,token_mask=self.mask,node_emb=self.names)
            self.native_text=self.native.texts([instruction]).detach().clone();self.appearance_text=self.native.texts([appearance_query(instruction)]).detach().clone()
            self.bounds=torch.tensor(self.record['world_bounds'],device=self.device,dtype=torch.float32)[None]
            self.world=self.bounds[:,None,:2]+self.inputs['grid_xy']*(self.bounds[:,None,2:]-self.bounds[:,None,:2])
            self.memory=CandidateMemory(self.world.shape[1],1024,self.device)
        features,valid,quality=self.visual.encode([state.current_observation],[self.world[0].cpu().numpy()])[0]
        features=features.to(self.device);valid=valid.to(self.device);quality=quality.to(self.device)
        history,history_valid,history_quality=self.memory.tensors()
        pose=torch.as_tensor(state.pose,device=self.device,dtype=torch.float32)[None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.model(static_inputs=self.inputs,tokens=self.tokens,token_mask=self.mask,node_emb=self.names,
                native_text=self.native_text,appearance_text=self.appearance_text,roi_features=features[None],roi_valid=valid[None],roi_quality=quality[None],
                pose_world=pose,bounds=self.bounds,ground_level=torch.tensor([GROUND_LEVEL[self.record['map_name']]],device=self.device),
                history_features=history[None],history_valid=history_valid[None],history_quality=history_quality[None],use_history=self.use_history,
                visual_control=self.visual_control,base_field=self.base)
        prediction=A3TrajectoryPolicy.predict_field(self,state,output.belief)
        frame=dict(step=state.step,pose=pose[0],roi_ids=torch.arange(len(features),device=self.device),roi_features=features,roi_valid=valid,roi_quality=quality)
        self.memory.update(frame)
        initial=retrieval_ids(output.static.belief_logits);dynamic=retrieval_ids(output.belief.belief_logits)
        winner=output.appearance_scores.masked_fill(~output.observed,-1e4).argmax(-1)
        prediction.diagnostics.update(policy='GraphNav-V1 A3 Epoch14 initialization + observed visual belief + causal quality memory + pretrained guarded control',
            observation_step=int(state.step),history_enabled=self.use_history,history_unique_views=self.memory.accepted,history_duplicates=self.memory.duplicates,history_unobserved_frames=self.memory.unobserved,
            visual_control=self.visual_control,residual_max_abs=float(output.delta.abs().max()),visual_gain=float(self.model.visual_gain.tanh()),
            static_top16_world_xy=self.world.gather(1,initial[...,None].expand(-1,-1,2))[0].cpu().tolist(),
            belief_top16_world_xy=self.world.gather(1,dynamic[...,None].expand(-1,-1,2))[0].cpu().tolist(),
            observed_grid_ids=output.observed[0].nonzero().flatten().cpu().tolist(),
            appearance_top1_world_xy=self.world[0,winner[0]].cpu().tolist()if output.observed.any()else None,
            active_observation=False)
        return prediction
