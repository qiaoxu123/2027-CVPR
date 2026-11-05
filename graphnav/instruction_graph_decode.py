"""Conservative trained span-parser output -> target/anchor path graph.

No GT, no RGB, no hidden target assignment. This decoder is useful only
AFTER the parser has been trained and calibrated on independent reviewed
data. It NEVER marks automatically predicted spatial relations as verified.
"""
from __future__ import annotations

from collections import deque

import torch

from .instruction_graph_parser_head import EDGE_CLASSES
from .instruction_relation_graph import (
    ROLES, FRAMES, compile_relation_graph,
)


@torch.no_grad()
def decode_span_graph(output, instruction: str, spans, public_bindings,
                      public_objects, world_bounds, *,
                      min_probability=0.5, max_hops=4,
                      world_north_verified=False):
    """Decode ONE sentence.

    spans: [(start_char,end_char), ...], aligned with output model indices.
    public_bindings: {span_index: {"public_ids":[...], "public_names":[...]}}
    obtained only from a public catalog/name matcher, never GT instance IDs.

    Returns a graph proposal, with ambiguities retained. Unknown target or
    unavailable anchored paths remain unresolved rather than force-filled.
    """
    if not (0 < min_probability <= 1) or max_hops < 1:
        raise ValueError("Invalid decoder controls")
    if output["role_logits"].shape[0]!=1 or output["edge_logits"].shape[0]!=1:
        raise ValueError("One sentence per decode")
    count=len(spans)
    if output["role_logits"].shape[1]<count:
        raise ValueError("Missing candidate span states")
    valid=output["valid_nodes"][0,:count].bool()
    roles=output["role_logits"][0,:count].softmax(-1)
    target_conf=roles[:,ROLES.index("target")].masked_fill(~valid,-1.)
    if count and valid.any():
        target_idx=int(target_conf.argmax())
        if float(target_conf[target_idx])<min_probability:
            target_idx=None
    else:
        target_idx=None
    entities=[];landmarks=[]
    for i,(a,b) in enumerate(spans):
        if not (0<=a<b<=len(instruction)):
            raise ValueError("Invalid candidate text span")
        role_id=int(roles[i].argmax())
        role=ROLES[role_id]
        public=public_bindings.get(i,{})
        if public.get("public_ids"):
            role="landmark"  # binding source is public map, not LLM hallucination
            landmarks.append(dict(start=a,end=b,text=instruction[a:b],
                                  public_ids=list(public["public_ids"]),
                                  public_names=list(public.get("public_names",[]))))
        entities.append(dict(id=f"e{i}",text=instruction[a:b],span=[a,b],
                             role="target" if i==target_idx else role,
                             attributes=[],public_names=list(public.get("public_names",[]))))
    edges=[];adj={i:[] for i in range(count)}
    probs=output["edge_logits"][0,:count,:count].softmax(-1)
    frames=output["frame_logits"][0,:count,:count].softmax(-1)
    pair_mask=output["valid_pairs"][0,:count,:count].bool()
    for i in range(count):
        for j in range(count):
            if not bool(pair_mask[i,j]):
                continue
            pred_id=int(probs[i,j].argmax())
            conf=float(probs[i,j,pred_id])
            if pred_id==0 or conf<min_probability:
                continue
            predicate=EDGE_CLASSES[pred_id]
            frame=FRAMES[int(frames[i,j].argmax())]
            index=len(edges)
            edges.append(dict(
                id=f"r{index}",subject=f"e{i}",object=f"e{j}",
                predicate=predicate,relation_kind="spatial_proposal",
                reference_frame=dict(frame=frame),
                source_span=[min(spans[i][0],spans[j][0]),
                             max(spans[i][1],spans[j][1])],
                source_text=instruction[min(spans[i][0],spans[j][0]):
                                        max(spans[i][1],spans[j][1])],
                prediction_confidence=conf,
                ambiguity=["learned_relation_not_human_verified"],
            ))
            adj[i].append((j,f"r{index}"))
    paths=[]
    if target_idx is not None:
        todo=deque([(target_idx,[target_idx],[])])
        while todo:
            node,chain,edge_ids=todo.popleft()
            if len(edge_ids)>max_hops:
                continue
            if node!=target_idx and public_bindings.get(node,{}).get("public_ids"):
                paths.append(dict(
                    entities=[f"e{x}" for x in chain],
                    relations=[dict(relation_id=k,traversal="forward") for k in edge_ids],
                    public_names=list(public_bindings[node].get("public_names",[])),
                    hops=len(edge_ids),ambiguity=["learned_path_not_human_verified"],
                ))
            if len(edge_ids)==max_hops:continue
            for next_node,relation_id in adj[node]:
                if next_node not in chain:
                    todo.append((next_node,chain+[next_node],edge_ids+[relation_id]))
    parsed=dict(
        original_text=instruction,map_name=None,target_GT_used=False,
        target=dict(entity=f"e{target_idx}" if target_idx is not None else None,
                    category=None,attributes=[]),
        entities=entities,relations=edges,landmarks=landmarks,
        direct_relations=[p for p in paths if p["hops"]==1],
        indirect_relations=[p for p in paths if p["hops"]>1],
        ambiguity=([] if target_idx is not None else ["target_role_unresolved"]),
    )
    graph=compile_relation_graph(
        parsed,public_objects,world_bounds,
        world_north_verified=world_north_verified
    )
    graph["schema"]="citynav-predicted-instruction-relation-graph-v1"
    graph["decoded_with_reviewed_parser"] = False
    graph["target_role_confidence"]=(
        float(target_conf[target_idx]) if target_idx is not None else None
    )
    graph["relation_threshold"]=min_probability
    graph["max_hops"]=max_hops
    return graph
