"""Conservative, GT-free direct cardinal predicate binding for Unified V2.

Explicit relation IDs: 0 unknown, 1 east, 2 west, 3 north, 4 south.
This fits the EXISTING 8-entry CandidateAnchorRelation predicate embedding;
no model weights or architecture need resizing. No implicit world frame for
'behind', 'left', 'right', compound relations or indirect intermediate scopes.

TRAIN: target GT is read ONLY by angular_consistency_loss AFTER forward.
RUNTIME: metadata binding depends only on original instruction and all
PUBLIC CityRefer polygons, never target location, nearest GT, or trajectory.

World XY is the CityNav map's coordinate axes. Geographic north/yaw
requires a separately verified CRS; this loss does not assume one.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from .candidate_geometry import FullCandidateDataset
from .semantic_geometry import canonical

PREDICATE_IDS={"east":1,"west":2,"north":3,"south":4}
FORBIDDEN_PREFIX={
    "in","inside","within","at","on","of","behind","near","by",
    "beside","between","across","along","opposite","around",
    "from","after","before","outside","over","under","toward",
    "towards","next","through","past","throughout","to",
}


def _unique_public_by_name(objects: dict):
    """Name -> physical object IDs, based solely on the full PUBLIC catalog."""
    names={}
    for oid,item in objects.items():
        name=canonical(item.get("name",""))
        if name:
            names.setdefault(name,set()).add(str(oid))
    return names


def direct_cardinal_reference(instruction: str, reference: dict, public_names: dict):
    """Return 1..4 only for a conservative, unambiguous *direct* phrase.

    Example accepted: 'the blue car north of One Stop'.
    Rejected: 'car in the parking lot north of One Stop' (indirect);
              'car behind One Stop' (unknown frame);
              two polygons named One Stop (identity ambiguous).
    No target GT enters this function.
    """
    name=canonical(reference.get("name",""))
    ref_id=str(reference.get("id",""))
    if not name or not ref_id or public_names.get(name)!= {ref_id}:
        return 0
    sentence=canonical(instruction)
    pattern=rf"\b(north|south|east|west)\s+of\s+(?:the\s+)?{re.escape(name)}$"
    match=re.search(pattern,sentence)
    if not match:
        return 0
    prefix=sentence[:match.start()].split()
    if not prefix or len(prefix)>22 or any(w in FORBIDDEN_PREFIX for w in prefix):
        return 0
    return PREDICATE_IDS[match.group(1)]


def predicate_ids_for_record(record, public_map, capacity,*, enabled=True):
    """Return (ids, reference_idx), where idx=-1 means NO angle label.

    Deliberately require one observed reference AND unique name/ID globally
    within the public map. This is an accuracy-first diagnostic subset; it is
    NOT full direct-relation recall. Everything else still trains Goal CE.
    """
    if capacity<1:raise ValueError("Missing entity capacity")
    ids=torch.zeros(capacity,dtype=torch.long)
    if not enabled:return ids,-1
    refs=record.get("landmarks",[])
    if len(refs)!=1:
        return ids,-1
    name_index=_unique_public_by_name(public_map)
    predicate=direct_cardinal_reference(record["instruction"],refs[0],name_index)
    if predicate:
        ids[0]=predicate
        return ids,0
    return ids,-1


class DirectBearingDataset(FullCandidateDataset):
    def __init__(self,*args, public_objects, enable_explicit_predicates=True,**kwargs):
        super().__init__(*args,**kwargs)
        public=json.loads(Path(public_objects).read_text(encoding="utf-8"))
        if not isinstance(public,dict):
            raise ValueError("Expected CityRefer map -> public objects catalog")
        self.public=public
        self.enable_explicit_predicates=bool(enable_explicit_predicates)
        for record in self.records:
            if record["map_name"] not in public:
                raise ValueError("Missing map from full public catalog: "+record["map_name"])
        self._predicates={}
        self._counts=Counter()
        for i,r in enumerate(self.records):
            ids,index=predicate_ids_for_record(
                r,public[r["map_name"]],self.capacity,
                enabled=self.enable_explicit_predicates)
            self._predicates[i]=(ids,index)
            self._counts["train_queries"]+=1
            self._counts["direct_cardinal_unique"]+=int(index>=0)
            self._counts["no_eligible_direct_cardinal"]+=int(index<0)

    @property
    def eligibility_audit(self):
        return dict(self._counts)

    def __getitem__(self,index):
        row=super().__getitem__(index)
        ids,anchor_index=self._predicates[int(index)]
        row["anchor_predicate_ids"]=ids
        row["angular_anchor_index"]=torch.tensor(anchor_index,dtype=torch.long)
        return row


def angular_consistency_loss(logits, pair_geometry, target_xy, nodes,
                             map_sizes_m, anchor_index, *,
                             bins=8, concentration=6., min_radius_m=1.):
    """Supervise the MAIN Belief's angular mass, not a weak auxiliary head.

    The target coordinate appears only HERE as TRAIN LABEL; cannot be used
    as a forward feature. Predicted bearing is induced by full [B,784] Belief.
    The direction target is a circular 8-bin soft label, avoiding wrap-around.

    Angular candidates use ONLY atan2 of EXISTING public candidate geometry.
    The module adds no XY offset or range to network inference.
    """
    if (logits.ndim!=2 or pair_geometry.ndim!=4 or
            pair_geometry.shape[:2]!=logits.shape or pair_geometry.shape[-1]!=10 or
            target_xy.shape!=(len(logits),2) or
            map_sizes_m.shape!=(len(logits),2) or
            anchor_index.shape!=(len(logits),) or nodes.ndim!=3 or
            nodes.shape[0]!=len(logits) or nodes.shape[-1]<2 or
            bins<4 or bins%2 or concentration<=0 or min_radius_m<0):
        raise ValueError("Incompatible angular supervision tensors")
    b,q=logits.shape
    if (anchor_index>=pair_geometry.shape[2]).any() or (anchor_index< -1).any():
        raise ValueError("Anchor indices out of bounds")
    rows=torch.arange(b,device=logits.device)
    safe=anchor_index.long().clamp_min(0)
    center_norm=nodes[rows,safe,:2].float()
    gt_delta=(target_xy.float()-center_norm)*map_sizes_m.float()
    gt_length=gt_delta.norm(dim=-1)
    valid=(anchor_index>=0)&(gt_length>min_radius_m)
    zero=logits.sum()*0.
    if not bool(valid.any()):
        return dict(loss=zero,valid=valid,angular_error_rad=zero.detach(),
                    pred_angle_bin=None,gt_angle_bin=None,
                    eligible_count=0)
    # [B,Q,2]: temporary candidate offsets only for atan2; radial size
    # does not enter the kernel/target prediction. Exclude undefined angle
    # at center of reference geometry.
    candidates=pair_geometry[rows,:,safe,:2].float()
    angles=torch.atan2(candidates[...,1],candidates[...,0])
    bins_angle=(torch.arange(bins,device=logits.device,dtype=torch.float32)
                * (2.*math.pi/bins))
    kernel=F.softmax(concentration*torch.cos(
        angles[:,:,None]-bins_angle[None,None,:]),dim=-1)
    magnitude=candidates.norm(dim=-1)
    cell_valid=(magnitude>min_radius_m/400.).float()
    belief=F.softmax(logits.float(),dim=-1)*cell_valid
    belief=belief/belief.sum(-1,keepdim=True).clamp_min(1e-9)
    induced=(belief[:,:,None]*kernel).sum(1).clamp_min(1e-9)
    true_angle=torch.atan2(gt_delta[:,1],gt_delta[:,0])
    target=F.softmax(concentration*torch.cos(
        true_angle[:,None]-bins_angle[None,:]),dim=-1)
    per_query=-(target*induced.log()).sum(-1)
    # Only eligibility based on public-name/direct-scope and centroid
    # undefinedness; NEVER choose target-nearest named polygon.
    loss=per_query[valid].mean()
    pred_bin=induced.argmax(-1)
    gt_bin=target.argmax(-1)
    deviation=torch.atan2(torch.sin(
        bins_angle[pred_bin]-true_angle),
        torch.cos(bins_angle[pred_bin]-true_angle)).abs()
    return dict(loss=loss,valid=valid,
                angular_error_rad=deviation[valid].mean().detach(),
                pred_angle_bin=pred_bin.detach(),
                gt_angle_bin=gt_bin.detach(),
                eligible_count=int(valid.sum().item()))
