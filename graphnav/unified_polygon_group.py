"""Public-only homonymous Polygon groups and set-level cardinal relation loss.

No target ID/GT, target distance, teacher trajectory or future RGB is used
for binding. A parsed public landmark name may correspond to 1..N Polygon
objects. We KEEP every original Polygon node and its own geometry and
unify ONLY the reference-identity constraint: direction from the target
to ANY group member can satisfy the literal cardinal relation.

This is an OR/max *over separate Polygon instances*. It never averages
their centroids, conflates disconnected footprints, or chooses the
target-nearest ground-truth Polygon. If a group cannot be held in the
existing fixed capacity, the relation is UNSUPPORTED rather than silently
truncated; all queries still enter Goal CE and the closed-loop denominator.

The loss uses cardinal direction named in ORIGINAL instruction. It is not
GT target yaw, and assumes map +X = east, +Y = north -- NOT verified
geographic north. The separate unique-anchor GT-angle experiment remains
in graphnav/unified_direct_bearing_loss.py.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .candidate_geometry import FullCandidateDataset
from .semantic_geometry import canonical
from .unified_direct_bearing_loss import FORBIDDEN_PREFIX, PREDICATE_IDS
from scripts.analyze_instruction_landmark_association import PublicNames


def public_cardinal_group(record, matcher, public_objects, capacity, *,
                          enabled=True):
    """Return a public-only, complete *set* of references named by 1 phrase.

    Parse only direct 'target north of PUBLIC_NAME' with one terminal
    public name. The prefix heuristic is deliberately conservative and
    MUST NOT be treated as a validated grammatical relation parser.

    A row may also have other unrelated named references. Only the group
    named in the terminal direct cardinal phrase gets predicate IDs.
    """
    ids=torch.zeros(capacity,dtype=torch.long)
    members=torch.zeros(capacity,dtype=torch.bool)
    if not enabled:return ids,members,"disabled"
    refs=record.get("landmarks",[])
    if not refs:return ids,members,"no_public_anchor"
    if len(refs)>capacity:
        return ids,members,"over_capacity"
    instruction=str(record["instruction"])
    map_name=record["map_name"]
    public_map=public_objects[map_name]
    matches=matcher.match(map_name,instruction)
    candidates=[]
    for mention in matches:
        if canonical(instruction[mention["end"]:]):
            continue  # The direct phrase must end the instruction.
        before=canonical(instruction[:mention["start"]])
        pattern=r"\b(north|south|east|west)\s+of\s+(?:the\s+)?$"
        found=re.search(pattern,before)
        if not found:continue
        prefix=before[:found.start()].split()
        if not prefix or len(prefix)>22 or any(w in FORBIDDEN_PREFIX for w in prefix):
            continue  # excludes 'car in parking lot north of X'
        matched_ids=set(str(oid) for oid in mention["public_ids"])
        selected=[(i,r) for i,r in enumerate(refs) if str(r.get("id","")) in matched_ids]
        if not selected:continue
        # No GT/nearest-Polygon choice. Every valid full-map matching instance
        # must be represented; a partial capped subset is not a valid group.
        if {str(r.get("id","")) for _,r in selected} != matched_ids:
            continue
        types={str(public_map[str(r["id"])].get("object_type","")).casefold()
               for _,r in selected}
        if len(types)!=1:
            continue  # same name across buildings/roads isn't one region
        candidates.append((found.group(1),[i for i,_ in selected]))
    if len(candidates)!=1:
        return ids,members,("no_conservative_direct_cardinal" if not candidates
                            else "multiple_direct_cardinal_phrases")
    direction,idxs=candidates[0]
    for i in idxs:
        ids[i]=PREDICATE_IDS[direction]
        members[i]=True
    return ids,members,("single_instance" if len(idxs)==1
                        else "same_name_multi_polygon")


def grouped_cardinal_loss(logits, pair_geometry, reference_mask,
                          predicate_ids, *,temperature=14.,
                          width_deg=45.,min_contrast=.08):
    """Belief must put mass in the angle sector of ANY named Polygon.

    Compatibility for candidate q is a permutation-invariant max over the
    separate public Polygon bearings:
        S(q)=max_j sigmoid(tau*[cos(phi(q,j)-theta_j)-cos(width)])
        L = -log(sum_q softmax(Belief(q))*S(q))

    Does NOT use target GT angle or distance to choose group member, and
    must not be mistaken for an independent held-out relation label.
    The residual gradient acts on the actual main [B,Q] Goal Belief.

    Distances are not model features of this loss: pair_geometry[...,0:2]
    are used only as temporary inputs to atan2.
    """
    if (logits.ndim!=2 or pair_geometry.ndim!=4 or
            pair_geometry.shape[:2]!=logits.shape or
            pair_geometry.shape[-1]!=10 or
            reference_mask.shape!=(len(logits),pair_geometry.shape[2]) or
            predicate_ids.shape!=reference_mask.shape or
            not 0<width_deg<90 or temperature<=0 or min_contrast<0):
        raise ValueError("Invalid cardinal/group tensor shapes")
    count=reference_mask.bool().sum(-1)
    active= count>0
    zero=logits.sum()*0.
    if not bool(active.any()):
        return dict(loss=zero,eligible_count=0,eligible=active,
                    supported_mass=torch.zeros(len(logits),device=logits.device),
                    angular_contrast=torch.zeros(len(logits),device=logits.device))
    # Current group has one literal direction ID for *every* member.
    ids=predicate_ids.long()
    if (ids[reference_mask.bool()]<1).any() or (ids[reference_mask.bool()]>4).any():
        raise ValueError("Missing cardinal predicate on an active public group")
    direction=torch.tensor([0.,0.,math.pi,math.pi/2,3*math.pi/2],
                           device=logits.device)
    phi=torch.atan2(pair_geometry[...,1].float(),pair_geometry[...,0].float())
    literal=direction[ids.clamp(0,4)][:,None,:]
    similarity=torch.cos(phi-literal)
    comp=torch.sigmoid(temperature*(similarity-
                          math.cos(math.radians(width_deg))))
    comp=comp.masked_fill(~reference_mask.bool()[:,None,:],0.)
    # Idempotent group OR: more same-name Polygon fragments do not
    # artificially amplify the compatibility merely by their count.
    satisfied=comp.amax(-1)
    contrast=satisfied.amax(-1)-satisfied.amin(-1)
    eligible=active&(contrast>=min_contrast)
    masses=(F.softmax(logits.float(),-1)*satisfied).sum(-1).clamp_min(1.e-8)
    if not bool(eligible.any()):
        return dict(loss=zero,eligible_count=0,eligible=eligible,
                    supported_mass=masses.detach(),angular_contrast=contrast.detach())
    return dict(
        loss=-masses[eligible].log().mean(),
        eligible_count=int(eligible.sum()),
        eligible=eligible,
        supported_mass=masses.detach(),
        angular_contrast=contrast.detach(),
    )


class PublicPolygonGroupDataset(FullCandidateDataset):
    """Exact PR23 public-blind references plus cardinal *group* metadata.

    All source rows must have been rebound through the public text matcher
    before use. This class NEVER reads annotations to select references.
    """
    def __init__(self,*args,public_objects,enable_predicates=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.public=json.loads(Path(public_objects).read_text(encoding="utf-8"))
        self.matcher=PublicNames(self.public)
        self.binding_audit=Counter()
        self._cardinals=[]
        for row in self.records:
            if row["map_name"] not in self.public:
                raise ValueError("PUBLIC catalog map missing")
            ids,mask,status=public_cardinal_group(
                row,self.matcher,self.public,self.capacity,
                enabled=enable_predicates)
            self._cardinals.append((ids,mask,status))
            self.binding_audit[status]+=1

    def __getitem__(self,index):
        item=super().__getitem__(index)
        ids,mask,_=self._cardinals[int(index)]
        item["anchor_predicate_ids"]=ids
        item["angular_group_mask"]=mask
        found=mask.nonzero().flatten()
        item["angular_anchor_index"]=torch.tensor(
            int(found[0]) if len(found)==1 else -1,dtype=torch.long)
        return item
