"""Weak role-to-public-Polygon supervision from text and PUBLIC geometry alone.

Labels are *sets* of candidate scene nodes. A uniquely named intermediate can
provide an exact public-ID set; an ordinary noun only provides a broad type
set. Neither operation can access target GT, target IDs or teacher bearings.
This is a noisy auxiliary task, NOT a certified instance-grounding label.
"""
from __future__ import annotations

from collections import Counter
import re

import torch
from .polygon_neighborhood_bearing import TYPE_TO_ID

# More specific nouns take precedence over generic road/ground words.
_TYPE_RULES = (
    (r"\b(?:parking\s+(?:lot|area|space)|car\s+park|parking)\b", "Parking"),
    (r"\b(?:footpath|sidewalk|pavement|pedestrian\s+path)\b", "Footpath"),
    (r"\b(?:bridge|overpass)\b", "Bridge"),
    (r"\b(?:car|cars|van|vans|truck|trucks|bus|buses|vehicle|vehicles|taxi|suv)\b", "Car"),
    (r"\b(?:building|buildings|house|houses|church|school|warehouse|tower|garage|office)\b", "Building"),
    (r"\b(?:road|street|lane|avenue|highway)\b", "TrafficRoad"),
    (r"\b(?:grass|lawn|field|garden|ground)\b", "Ground"),
    (r"\b(?:river|lake|water|canal)\b", "Water"),
)
_WEAK_RELATIONS = {"to", "of", "with", "associated region", "possessive region association"}


def role_type(text):
    """A textual category hint, not an instance identity or an angle label."""
    t = str(text).casefold()
    for pattern, kind in _TYPE_RULES:
        if re.search(pattern, t):
            return TYPE_TO_ID[kind]
    return None


def _aligned_step(path, k):
    """Verify edge subject/reference roles match the claimed traversal scope."""
    nodes = path.get("nodes", [])
    steps = path.get("steps", [])
    if len(nodes) != len(steps) + 1 or not 0 <= k < len(steps):
        return False
    left, right = nodes[k]["node"], nodes[k+1]["node"]
    edge = steps[k]
    subj, obj = edge["subject"]["node"], edge["object"]["node"]
    if {left, right} != {subj, obj}:
        return False
    if edge.get("orientation") != (1 if subj == left else -1):
        return False
    relation = str(edge.get("relation", "")).casefold().strip()
    if not relation or relation in _WEAK_RELATIONS or "unresolved" in relation:
        return False
    return True


def public_role_targets(parsed, scene, query, *, max_steps=4):
    """Create list of (anchor, hop, public candidate SET) labels.

    Mirrors the deterministic path choice in prepare_relation_bearing_features.
    All arguments contain only text parsing, public scene geometry and indices.
    No function argument for GT is present. If path scope is ambiguous, abstain.
    """
    ids = [str(v) for v in scene["node_ids"]]
    valid = scene["valid"].bool().cpu().tolist()
    types = scene["types"].long().cpu().tolist()
    anchors = query["anchor_indices"].long().cpu().tolist()
    anchor_valid = query["anchor_valid"].bool().cpu().tolist()
    if len(ids) > len(valid) or len(ids) > len(types):
        raise ValueError("Public node and geometry lengths disagree")
    results = []
    counts = Counter()
    for j, is_anchor in enumerate(anchor_valid):
        if not is_anchor:
            continue
        anchor_index = anchors[j]
        if anchor_index < 0 or anchor_index >= len(ids) or not valid[anchor_index]:
            raise ValueError("Invalid public anchor index")
        ref = ids[anchor_index]
        choices = [p for p in parsed.get("paths", [])
                   if p.get("semantic_confident", False) and
                   ref in {str(x) for x in p.get("public_ids", [])}]
        choices.sort(key=lambda p: (p["hops"], str(p["nodes"])))
        if not choices:
            counts["no_confident_path"] += 1
            continue
        choice = choices[0]
        hops = int(choice["hops"])
        if not 0 < hops <= max_steps:
            counts["unsupported_hop_count"] += 1
            continue
        # A second equally short but structurally different path is ambiguous.
        if len(choices) > 1 and choices[1]["hops"] == hops and (
                choices[1]["nodes"] != choice["nodes"] or
                choices[1]["steps"] != choice["steps"]):
            counts["ambiguous_path_scope"] += 1
            continue
        for t, k in enumerate(reversed(range(hops))):
            if not _aligned_step(choice, k):
                counts["invalid_or_weak_scope"] += 1
                continue
            entity = choice["nodes"][k]
            known = {str(x) for x in entity.get("public_ids", [])}
            if known:
                candidates = [n for n, oid in enumerate(ids) if oid in known and valid[n]]
                kind = "named_public"
                weight = 1.0
            elif entity.get("kind") == "ordinary_object":
                type_id = role_type(entity.get("text", ""))
                candidates = [n for n in range(len(ids))
                              if valid[n] and n != anchor_index and types[n] == type_id] if type_id is not None else []
                kind = "typed_intermediate" if k > 0 else "typed_target"
                weight = 0.25 if k > 0 else 0.10
            else:
                candidates = []
                kind = "unknown"
                weight = 0.0
            if not candidates:
                counts["unmapped_role"] += 1
                continue
            if len(candidates) >= sum(valid):
                counts["uninformative_full_scene"] += 1
                continue
            results.append(dict(anchor=j, step=t, candidates=candidates,
                                kind=kind, weight=weight,
                                role_text=entity.get("text", "")))
            counts[kind] += 1
    return results, dict(counts)


def role_set_loss(step_probabilities, batch_labels):
    """Query-equal marginal NLL over possible public role bindings.

    For generic nouns, maximize probability mass on *any* matching public
    category rather than inventing one correct instance. Zero labels -> zero
    differentiable loss. The model's angle GT stays in a separate objective.
    """
    if step_probabilities.ndim != 4 or len(batch_labels) != step_probabilities.shape[0]:
        raise ValueError("Expected [B,anchors,hops,nodes] and B weak-label lists")
    b, r, h, n = step_probabilities.shape
    losses = []
    for i, labels in enumerate(batch_labels):
        parts, weights = [], []
        for label in labels:
            a, t = int(label["anchor"]), int(label["step"])
            nodes = list(label["candidates"])
            if not (0 <= a < r and 0 <= t < h and nodes and
                    all(0 <= k < n for k in nodes)):
                raise ValueError("Public role supervision index out of range")
            weight = float(label["weight"])
            if not 0 < weight <= 1:
                raise ValueError("Role supervision weight out of range")
            # Multiple candidates = legitimate ambiguity, not a GT selection.
            mass = step_probabilities[i, a, t, nodes].sum().clamp_min(1e-9)
            parts.append(-mass.log() * weight)
            weights.append(weight)
        if parts:
            losses.append(torch.stack(parts).sum() / sum(weights))
    if not losses:
        return step_probabilities.sum() * 0, 0
    return torch.stack(losses).mean(), len(losses)
