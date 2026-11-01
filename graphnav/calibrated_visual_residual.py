"""Observed RGB evidence correction conditioned on a frozen geometric belief.

Labels are accepted only by loss functions, never by forward.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .full_localization import GoalField
from .stage1 import preserved_proposal_order


class CalibratedVisualResidual(nn.Module):
    def __init__(self,context_dim=128,vision_dim=1024,hidden=64,bound=2.,centered=True,use_pose=True,relation=False,relation_attention=False,structured_relation=False,official_relation=False):
        super().__init__();self.bound=float(bound);self.centered=bool(centered);self.use_pose=bool(use_pose)
        self.image=nn.Linear(vision_dim,32,bias=False)
        self.text=nn.Linear(vision_dim,32,bias=False)
        self.geometry=nn.Linear(context_dim,16)
        shared_rng=torch.random.get_rng_state()
        self.official_relation=None
        if official_relation:
            if relation or relation_attention or structured_relation:raise ValueError('Official pair adapter is a separate relation experiment')
            from .aerialvg_pair_residual import AerialVGPairResidual
            self.official_relation=AerialVGPairResidual(context_dim=context_dim,vision_dim=vision_dim)
        self.relation=nn.Sequential(nn.Linear(32+3,32),nn.GELU(),nn.Linear(32,32)) if relation else None
        if relation_attention and not relation:raise ValueError('Relation attention requires relation=True')
        self.relation_attention=nn.Sequential(nn.Linear(32*3+16+3,32),nn.GELU(),nn.Linear(32,1)) if relation_attention else None
        if relation_attention:
            self.reference_query=nn.Linear(768,32,bias=False)
            self.reference_key=nn.Linear(768,32,bias=False)
        self.structured_relation=bool(structured_relation)
        if structured_relation:
            if not relation_attention:raise ValueError('Structured roles require relation attention')
            self.role_target=nn.Linear(768,32,bias=False)
            self.role_relation=nn.Linear(768,16,bias=False)
            self.role_score=nn.Sequential(nn.Linear(85,32),nn.GELU(),nn.Linear(32,1))
            nn.init.zeros_(self.role_score[-1].weight);nn.init.zeros_(self.role_score[-1].bias)
        # Optional branches must not change the initialization of shared layers.
        torch.random.set_rng_state(shared_rng)
        self.network=nn.Sequential(nn.Linear(16+32*2+18+8,hidden),nn.GELU(),nn.Linear(hidden,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():self.network[-1].bias[1]=-1.

    def forward(self,base,public,static_inputs,tokens=None,token_mask=None,node_emb=None):
        images=public['roi_features'].float();past=public['history_features'].float()
        valid=public['roi_valid'];remembered=public['history_valid']
        if self.relation is not None or self.official_relation is not None:
            # A flagged region with no usable features/quality is not a relation anchor.
            valid=valid & (images.abs().sum(-1)>0) & (public['roi_quality'][...,0]>0) & (public['roi_quality'][...,1]>0)
            remembered=remembered & (past.abs().sum(-1)>0) & (public['history_quality'][...,0]>0) & (public['history_quality'][...,1]>0)
        known=valid.any(-1)|remembered.any(-1)
        meaningful=(images.abs().sum((-1,-2))>0)|(past.abs().sum((-1,-2))>0)
        known=known&meaningful
        def pool(values,mask,quality):
            weight=(quality[...,0].clamp(0,1)*quality[...,1].clamp(0,1))*mask
            weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-9)
            return (values*weight[...,None]).sum(-2)
        current=pool(images,valid,public['roi_quality']);history=pool(past,remembered,public['history_quality'])
        language=F.normalize(public['native_text'].float()+public['appearance_text'].float(),dim=-1)
        instruction=self.text(language)[:,None]
        rgb_current=self.image(current)*instruction;rgb_history=self.image(history)*instruction
        role_raw=torch.zeros_like(base.static.belief_logits)
        if self.relation is not None:
            # Sparse observed anchors keep memory O(cells * 16), including dense grids.
            xy=static_inputs['grid_xy'].float()
            ids=base.static.belief_logits.float().masked_fill(~known,-1e9).topk(min(16,known.shape[1]),dim=1).indices
            if self.relation_attention is not None:
                # Greedy spatial diversity prevents many adjacent cells from monopolizing anchors.
                score=base.static.belief_logits.float().masked_fill(~known,-1e9).clone()
                reference_bias=torch.zeros_like(score)
                if node_emb is not None and tokens is not None:
                    phrase=(tokens.float()*token_mask[...,None]).sum(1)/token_mask.sum(1,keepdim=True).clamp_min(1)
                    node_score=(self.reference_query(phrase)[:,None]*self.reference_key(node_emb.float())).sum(-1)/(32**.5)
                    node_valid=static_inputs['node_valid'].bool()
                    weight=node_score.float().masked_fill(~node_valid,-1e4).softmax(-1)*node_valid
                    weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-9)
                    if self.structured_relation:
                        bound=public['role_valid']
                        index=public['role_reference_index'].clamp_min(0)
                        exact=F.one_hot(index,num_classes=weight.shape[-1]).to(weight.dtype)*node_valid
                        weight=torch.where(bound[:,None],exact,weight)
                        relative=(static_inputs['pair_geometry'].float()[...,:2]*weight[:,None,:,None]).sum(-2)
                        direction=public['role_direction'].float()
                        compatibility=(relative*direction[:,None]).sum(-1,keepdim=True)
                        target=self.role_target(public['role_target'].float())[:,None]
                        ref=self.reference_query(public['role_reference'].float())[:,None].expand(-1,relative.shape[1],-1)
                        rel=self.role_relation(public['role_relation'].float())[:,None].expand(-1,relative.shape[1],-1)
                        role_visual=self.image(current+history)*target
                        role_features=torch.cat((role_visual,ref,rel,relative,compatibility,direction[:,None].expand(-1,relative.shape[1],-1)),-1)
                        role_raw=self.role_score(role_features).squeeze(-1)*bound[:,None]
                    # Public contour distance associates observed cells with named map regions;
                    # it is not a claimed visual instance identity match.
                    boundary=static_inputs['pair_geometry'].float()[...,5].abs()
                    reference_bias=(torch.exp(-boundary/.1)*weight[:,None]).sum(-1)
                    score=score+reference_bias
                selected=[];anchor_valid=[]
                for _ in range(min(16,known.shape[1])):
                    index=score.argmax(-1);selected.append(index);anchor_valid.append(score.amax(-1)>-1e8)
                    position=xy.gather(1,index[:,None,None].expand(-1,1,2))
                    nearby=(xy-position).norm(dim=-1)<.08
                    score=score.masked_fill(nearby,-1e9)
                    score.scatter_(1,torch.stack(selected,1),-1e9)
                ids=torch.stack(selected,1)
            visual=(rgb_current+rgb_history)*known[...,None]
            anchors=visual.gather(1,ids[...,None].expand(-1,-1,32))
            positions=xy.gather(1,ids[...,None].expand(-1,-1,2))
            offset=positions[:,None]-xy[:,:,None]
            edges=torch.cat((anchors[:,None]-visual[:,:,None],offset,offset.norm(dim=-1,keepdim=True)),dim=-1)
            support=known.gather(1,ids)[:,None].expand(-1,known.shape[1],-1)
            if self.relation_attention is not None:support=support & torch.stack(anchor_valid,1)[:,None]
            support=support & (ids[:,None]!=torch.arange(known.shape[1],device=ids.device)[None,:,None])
            messages=self.relation(edges)*support[...,None]
            if self.relation_attention is None:
                context=messages.sum(-2)/support.sum(-1,keepdim=True).clamp_min(1)
            else:
                geom=self.geometry(base.static.dense_context.float())
                shape=edges.shape[:3]
                attention_input=torch.cat((visual[:,:,None].expand(*shape,32),anchors[:,None].expand(*shape,32),instruction[:,:,None].expand(*shape,32),geom[:,:,None].expand(*shape,16),edges[...,-3:]),-1)
                scores=self.relation_attention(attention_input).squeeze(-1).float()
                scores=scores+reference_bias.gather(1,ids)[:,None]
                weights=scores.masked_fill(~support,-1e4).softmax(-1)*support
                weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-9)
                context=(messages*weights[...,None]).sum(-2)
            rgb_current=rgb_current+context*known[...,None]
        logits=base.static.belief_logits.float();relative=(logits-logits.amax(-1,keepdim=True)).clamp(-20,0)/5
        probability=logits.softmax(-1)*(logits.shape[-1]/784.)
        xy=static_inputs['grid_xy'].float();bounds=public['bounds'].float()
        pose=(public['pose_world'][:,:2]-bounds[:,:2])/(bounds[:,2:]-bounds[:,:2])
        spatial=(xy-pose[:,None]).norm(dim=-1) if self.use_pose else F.cosine_similarity(current,language[:,None],dim=-1)
        features=torch.stack((relative,probability,base.current_scores.float()/20,base.appearance_scores.float()/20,
            valid.any(-1).float(),remembered.any(-1).float(),spatial,known.float()),-1)
        quality=torch.cat((public['roi_quality'].float().flatten(-2),public['history_quality'].float().flatten(-2)),-1)
        x=torch.cat((self.geometry(base.static.dense_context.float()),rgb_current,rgb_history,quality,features),-1)
        raw,confidence=self.network(x).float().unbind(-1)
        raw=raw+role_raw.float()
        if self.official_relation is not None:
            raw=raw+self.official_relation(current,history,base.static.dense_context.float(),xy,tokens,token_mask,known,logits)
        centered=raw-(raw*known).sum(-1,keepdim=True)/known.sum(-1,keepdim=True).clamp_min(1) if self.centered else raw
        delta=self.bound*centered.tanh()*confidence.sigmoid()*known
        return delta,confidence.sigmoid(),known


def corrected_field(static,delta,grid_xy):
    if not delta.detach().any():return static
    logits=static.belief_logits.float()+delta
    ids=preserved_proposal_order(logits,static.candidate_ids)
    context=static.dense_context
    return GoalField(logits,ids,grid_xy.gather(1,ids[...,None].expand(-1,-1,2)),
        context.gather(1,ids[...,None].expand(-1,-1,context.shape[-1])),dense_context=context,
        visual_coverage=(delta!=0).float())


def correction_loss(logits,delta,grid_world,target_world,known,rank_weight=0.):
    distance=(grid_world-target_world[:,None,:2]).norm(dim=-1)
    legal=distance<=20.
    # If a coarse grid has no cell in 20m, retain the nearest cell for a finite loss.
    nearest=distance.argmin(-1)
    legal=legal.clone();legal[torch.arange(len(legal),device=legal.device),nearest]|=~legal.any(-1)
    updated=logits.float()+delta
    mass=torch.logsumexp(updated,-1)-torch.logsumexp(updated.masked_fill(~legal,-1e4),-1)
    old=torch.logsumexp(logits.float(),-1)-torch.logsumexp(logits.float().masked_fill(~legal,-1e4),-1)
    drift=(delta.square()*known).sum(-1)/known.sum(-1).clamp_min(1)
    good=updated.masked_fill(~legal,-1e4).amax(-1)
    bad=updated.masked_fill(legal,-1e4).amax(-1)
    ranking=F.softplus((bad-good)/.25).mean()*.25
    total=mass.mean()+.1*F.relu(mass-old).mean()+.02*drift.mean()+rank_weight*ranking
    return dict(total=total,mass=mass.mean(),baseline_mass=old.mean(),drift=drift.mean(),ranking=ranking)


def observed_hard_ranking(logits,delta,world,target,known,visual):
    """Train-only target supervision; compare observed, visually similar distractors."""
    legal=(world-target[:,None,:2]).norm(dim=-1)<=20
    positive=legal & known;negative=(~legal) & known
    eligible=positive.any(-1)&negative.any(-1)
    ids=logits.float().masked_fill(~positive,-1e9).argmax(-1)
    reference=visual.gather(1,ids[:,None,None].expand(-1,1,visual.shape[-1]))
    similarity=F.cosine_similarity(visual,reference,dim=-1).detach()
    hard_ids=similarity.masked_fill(~negative,-1e9).topk(min(8,visual.shape[1]),dim=-1).indices
    scores=logits.float()+delta
    bad=scores.masked_fill(~negative,-1e4).gather(1,hard_ids).amax(-1)
    good=scores.masked_fill(~positive,-1e4).amax(-1)
    losses=F.softplus((.2+bad-good)/.25)*.25
    return (losses*eligible).sum()/eligible.sum().clamp_min(1)
