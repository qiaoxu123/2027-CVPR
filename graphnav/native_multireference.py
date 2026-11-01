"""Public multi-landmark geometry; no target instance annotations are invented."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .native_relation_grounding import public_relation_descriptors
from .semantic_geometry import SpatialRelation, RELATIONS, build_entities, canonical, parse_spatial_relations

CAPACITY = 8


def full_public_graph(static, index):
    record = static.records[index]
    # Do not inherit the original model's entity truncation; do not recompute A0.
    entities = build_entities(record.get('landmarks', []), [], record['world_bounds'])
    entities += [e for e in static._prepared(index)[0]['entities'] if e['kind']=='road']
    unique, seen = [], set()
    for entity in entities:
        key = (canonical(entity['name']), entity['kind'], np.asarray(entity['geometry_m']).tobytes())
        if key not in seen:
            unique.append(entity)
            seen.add(key)
    return dict(entities=unique, relations=parse_spatial_relations(record['instruction'], unique))


def multireference_descriptors(world_xy, graph):
    """Keep distinct true anchors plus nearby actual map distractors.

    Binding flags encode explicit public instruction-name correspondence;
    distractors are not labeled target-negative visual instances.
    """
    entities = graph['entities']
    slots = []
    for relation in graph['relations']:
        if not relation.anchors or any(len(a)!=1 for a in relation.anchors):
            continue
        for anchor in relation.anchors:
            pair = (anchor[0], relation.kind)
            if pair not in slots:
                slots.append(pair)
    if len(slots)>CAPACITY:
        raise ValueError('Too many real reference anchors; do not silently truncate')
    positive_count = len(slots)
    positive_ids = {i for i, _ in slots}
    if slots:
        center = np.mean([np.asarray(entities[i]['geometry_m']).mean(0) for i in positive_ids], axis=0)
        others = [i for i, e in enumerate(entities) if i not in positive_ids and e['name']]
        others.sort(key=lambda i:(float(np.linalg.norm(np.asarray(entities[i]['geometry_m']).mean(0)-center)), i))
        slots += [(i, slots[0][1]) for i in others[:min(3, CAPACITY-len(slots))]]
    features = torch.zeros(len(world_xy), CAPACITY, 9)
    kinds = torch.zeros(CAPACITY, dtype=torch.long)
    valid, binding = torch.zeros(CAPACITY, dtype=torch.bool), torch.zeros(CAPACITY, dtype=torch.bool)
    for j,(index,kind) in enumerate(slots):
        # A between relation keeps TWO anchor positions; its per-anchor distance
        # descriptor uses near geometry while preserving the between type.
        descriptor_kind = 'near' if kind=='between' else kind
        mini = dict(entities=entities, relations=[SpatialRelation(descriptor_kind, ((index,),))])
        f,_,_ = public_relation_descriptors(world_xy, mini)
        features[:,j,:8] = f[:,0]
        features[:,j,8] = float(j<positive_count)
        kinds[j],valid[j],binding[j] = RELATIONS.index(kind),True,j<positive_count
    return dict(reference_features=features.half(), reference_types=kinds,
                reference_valid=valid, reference_binding=binding)


class MultiReferenceRanker(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.candidate = nn.Sequential(nn.Linear(520,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.reference = nn.Sequential(nn.Linear(9,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.kind = nn.Embedding(len(RELATIONS),hidden)
        self.query,self.key,self.value = [nn.Linear(hidden,hidden,bias=False) for _ in range(3)]
        self.head = nn.Sequential(nn.Linear(hidden*3,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.geometry_head = nn.Linear(hidden,4)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, text, text_mask, geometry, valid,
                reference_features, reference_types, reference_valid, native_logits,
                *, enabled=True):
        text = text.float().masked_fill(~text_mask.bool()[...,None],0.)
        if not all(torch.isfinite(v).all() for v in [features,text,geometry,reference_features,native_logits]):
            raise ValueError('Nonfinite public reference input')
        sentence = text.sum(1)/text_mask.sum(1,keepdim=True).clamp_min(1)
        q = self.candidate(torch.cat([features.float(),sentence[:,None].expand_as(features),geometry.float()],-1))
        ref = self.reference(reference_features.float())+self.kind(reference_types)[:,None]
        mask = reference_valid[:,None] & valid[:,:,None] & enabled
        scores = (self.query(q)[:,:,None]*self.key(ref)).sum(-1)/math.sqrt(q.shape[-1])
        attention = scores.masked_fill(~mask,-1e4).softmax(-1)*mask
        attention = attention/attention.sum(-1,keepdim=True).clamp_min(1e-9)
        context = (attention[...,None]*self.value(ref)).sum(2)
        delta = 3*torch.tanh(self.head(torch.cat([q,context,q*context],-1)).squeeze(-1))
        has_relation = reference_valid.any(-1) & enabled
        delta = torch.where(has_relation[:,None] & valid, delta, torch.zeros_like(delta))
        return dict(logits=(native_logits.float()+delta).masked_fill(~valid,-1e4),
                    delta=delta,attention=attention,geometry_prediction=self.geometry_head(ref),
                    has_relation=has_relation)


def auxiliary_reference_loss(prediction, x):
    eligible = x['valid'][:,:,None] & x['reference_valid'][:,None]
    binding = x['reference_binding'].float()
    target = binding/binding.sum(-1,keepdim=True).clamp_min(1)
    row = x['valid'] & x['reference_binding'].any(-1)[:,None]
    cross_entropy = -(prediction['attention'].clamp_min(1e-9).log()*target[:,None]).sum(-1)
    attention_loss = cross_entropy[row].mean() if row.any() else cross_entropy.sum()*0
    target_geometry = x['reference_features'].float()[...,[1,2,3,7]]
    geometry_loss = F.smooth_l1_loss(prediction['geometry_prediction'][eligible],target_geometry[eligible]) if eligible.any() else prediction['geometry_prediction'].sum()*0
    return attention_loss,geometry_loss


def belief_training_loss(updated, native, x, has_relation):
    """Final grid utility plus KL and correct-Native margin preservation."""
    distance = (x['grid_world']-x['target_world'][:,None,:2]).norm(dim=-1)
    positive = distance<=20
    rows = has_relation & positive.any(-1)
    logp = updated.log_softmax(-1)
    per = -torch.logsumexp(logp.masked_fill(~positive,-1e4),-1)
    localization = per[rows].mean() if rows.any() else per.sum()*0
    kl = F.kl_div(logp,native.softmax(-1).detach(),reduction='none').sum(-1)
    preservation = kl[has_relation].mean() if has_relation.any() else kl.sum()*0
    native_index = native.argmax(-1)
    native_correct = distance.gather(1,native_index[:,None]).squeeze(-1)<=20
    safe = has_relation & native_correct
    margin = torch.relu(updated.masked_fill(positive,-1e4).max(-1).values-
                        updated.masked_fill(~positive,-1e4).max(-1).values+.1)
    safe_loss = margin[safe].mean() if safe.any() else margin.sum()*0
    return localization,preservation,safe_loss
