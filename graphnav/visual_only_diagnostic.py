"""Opt-in original visual branch with no A3 or named-polygon localization."""
import torch
from torch.nn import functional as F
from .full_localization import GoalField
from .stage1 import preserved_proposal_order


def visual_only_field(scores,observed,grid_uv,*,hidden,top_k,visual_gain,residual_bound):
    """Use the original residual exactly on a spatially uniform prior.

    No unobserved cell is assigned absence evidence. Neutral zero controller
    context explicitly removes A3 context; it is not a fabricated RGB feature.
    """
    centered=scores-(scores*observed).sum(-1,keepdim=True)/observed.sum(-1,keepdim=True).clamp_min(1)
    delta=residual_bound*torch.tanh(visual_gain)*torch.tanh(centered/5.)*observed
    neutral=torch.zeros_like(delta)
    base_ids=neutral.topk(top_k,dim=-1).indices
    ids=preserved_proposal_order(delta,base_ids)
    dense=torch.zeros(*scores.shape,hidden,device=scores.device)
    return GoalField(delta.float(),ids,grid_uv.gather(1,ids[...,None].expand(-1,-1,2)),
        dense.gather(1,ids[...,None].expand(-1,-1,hidden)),
        visual_coverage=observed.float(),dense_context=dense)


class VisualOnlyLandmarkPolicy:
    """Shares loaded frozen components, but cannot run geometry localization."""
    def __init__(self,original):
        self.original=original
        self.device=original.device
        self.head=original.head
        self.stop_threshold=original.stop_threshold
        self.stop_positive_weight=original.stop_positive_weight
        self.execution=original.execution
        self.explore_candidates=original.explore_candidates
        self.geometry_calls=0
        def forbid_geometry(module,args):
            self.geometry_calls+=1
            raise AssertionError('Geometry branch must never run in visual-only diagnostic')
        self.guard=original.model.a3.register_forward_pre_hook(forbid_geometry)

    def set_episode(self,episode,transform):
        if episode.references or episode.reference_names:
            raise ValueError('Visual-only policy must not receive landmark geometry/name bindings')
        from .geometry import grid_centers
        from .stage1_data import CandidateMemory
        from .evidence_grounding import appearance_query
        self.record=dict(map_name=episode.map_name,instruction=episode.instruction,
            world_bounds=[transform.x_min,transform.y_min,transform.x_max,transform.y_max])
        self.bounds=torch.tensor(self.record['world_bounds'],device=self.device,dtype=torch.float32)[None]
        self.uv=torch.as_tensor(grid_centers(self.original.args.grid_size),device=self.device)[None]
        self.world=self.bounds[:,None,:2]+self.uv*(self.bounds[:,None,2:]-self.bounds[:,None,:2])
        self.memory=CandidateMemory(self.world.shape[1],1024,self.device)
        self.native_text=self.original.native.texts([episode.instruction]).detach().clone()
        self.appearance_text=self.original.native.texts([appearance_query(episode.instruction)]).detach().clone()
        self.visited_goal_cells=set()

    @torch.no_grad()
    def predict(self,state,instruction,*,oracle_goal=None):
        if instruction!=self.record['instruction']:raise ValueError('Unexpected language mutation')
        o=self.original
        features,valid,quality=o.visual.encode([state.current_observation],[self.world[0].cpu().numpy()])[0]
        features=features.to(self.device);valid=valid.to(self.device);quality=quality.to(self.device)
        past,past_valid,past_quality=self.memory.tensors()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            text=F.normalize(self.native_text.float()+self.appearance_text.float(),dim=-1)
            current,known,_=o.model.matcher(text,features[None],valid[None],quality[None])
            scores,observed=current,known
            if o.use_history:
                old,_,_=o.model.matcher(text,past[None],past_valid[None],past_quality[None])
                scores,observed,_=o.model.history(current,valid[None],quality[None],old,past_valid[None],past_quality[None])
            field=visual_only_field(scores,observed,self.uv,hidden=o.model.a3.hidden,top_k=o.model.a3.top_k,
                visual_gain=o.model.visual_gain,residual_bound=o.model.residual_bound)
            if oracle_goal is not None:
                from .landmark_navigation_diagnostic import oracle_goal_field
                field=oracle_goal_field(field,self.uv,self.bounds,oracle_goal,diagnostic=True)
        from .a3_runtime import A3TrajectoryPolicy
        prediction=A3TrajectoryPolicy.predict_field(self,state,field)
        pose=torch.as_tensor(state.pose,device=self.device)
        self.memory.update(dict(step=state.step,pose=pose,roi_ids=torch.arange(len(features),device=self.device),
            roi_features=features,roi_valid=valid,roi_quality=quality))
        prediction.diagnostics.update(geometry_branch_calls=self.geometry_calls,geometry_prior='uniform',
            candidate_context='neutral_zero_no_A3',observed_cells=int(observed.sum()),
            visual_score_range=float((scores.max()-scores.min()).float()),
            visual_logit_range=float((field.belief_logits.max()-field.belief_logits.min()).float()),
            oracle_control=oracle_goal is not None,history_unique_views=self.memory.accepted,
            belief_top1_xy=self.world[0,field.belief_logits.argmax(-1)[0]].cpu().tolist())
        return prediction
