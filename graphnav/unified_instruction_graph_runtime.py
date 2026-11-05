"""Opt-in real-navigation Unified V2 + weak instruction graph residual policy.

Baseline inference remains complete-instruction UnifiedPGVBFullRelationPolicy.
Graph branch adds bounded candidate-specific logits AFTER trained Geometry and
Relation, BEFORE the unchanged StableGoal + Trajectory/Stop controller.
All public polygons originate from the original runtime record. An auxiliary
map of their public IDs is built for graph binding without modifying inputs.

Until reliable geometric-frame supervision exists, do NOT infer that a weak
parser graph certifies any direct/indirect relation. Requires spaCy model.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import torch

from .a3_runtime import A3TrajectoryPolicy
from .checkpointing import file_sha256
from .instruction_relation_graph import compile_relation_graph
from .landmark_alias import resolve_references
from .unified_full_relation_runtime import UnifiedPGVBFullRelationPolicy
from .unified_instruction_graph_belief import UnifiedInstructionGraphBelief
from .unified_instruction_graph_data import empty_graph,graph_input_batch


def attach_public_ids(record,episode,catalog,max_landmarks):
    """Restore separately the public IDs omitted by original runtime records.

    Never mutate record/model static_inputs: doing so would change the old
    baseline, breaking our paired causal comparison.
    """
    public_objects=catalog.objects[episode.map_name]
    names=episode.reference_names or tuple(ref.name for ref in episode.references)
    refs=resolve_references(names,public_objects,max_landmarks=max_landmarks,
                            match_mode="fuzzy")
    if len(refs)!=len(record["landmarks"]):
        raise ValueError("Public-ID binding does not match original static input")
    attached=[]
    for given,found in zip(record["landmarks"],refs):
        if (given["name"]!=found["name"] or
                given["object_type"]!=found["object_type"]):
            raise ValueError("Public reference order unexpectedly changed")
        attached.append(dict(given,id=str(found["id"])))
    return dict(record,landmarks=attached)


def replace_goal_field_logits(field,grid_xy,logits):
    """Re-rank BOTH Top-K candidate IDs and their context with new Belief."""
    if field.dense_context is None:
        raise ValueError("Graph runtime requires dense candidate context")
    if logits.shape!=field.belief_logits.shape:
        raise ValueError("Graph Belief has mismatched grid")
    k=field.candidate_ids.shape[1]
    top=logits.topk(k,-1).indices
    dense=field.dense_context
    xy=grid_xy.gather(1,top[...,None].expand(-1,-1,2))
    context=dense.gather(1,top[...,None].expand(-1,-1,dense.shape[-1]))
    return replace(field,belief_logits=logits,candidate_ids=top,
                   candidate_xy=xy,candidate_context=context)


class UnifiedGraphBeliefPolicy(UnifiedPGVBFullRelationPolicy):
    def __init__(self,stage1,original_v2,trained,directory,*,
                 graph_head_checkpoint,device="cuda",
                 relation_control="full",visual_control="off",
                 geometry_source="trained"):
        if relation_control!="full" or visual_control!="off" or geometry_source!="trained":
            raise ValueError("Initial graph experiment needs trained full Geometry+Relation, RGB OFF")
        super().__init__(stage1,original_v2,trained,directory,device=device,
                         relation_control=relation_control,
                         visual_control=visual_control,
                         geometry_source=geometry_source)
        state=torch.load(graph_head_checkpoint,map_location="cpu",weights_only=True)
        if (state.get("format")!="unified-v2-graph-belief-residual-v1"
                or state.get("epoch",0)!=state.get("manifest",{}).get("epochs")
                or state["manifest"].get("max_batches")!=0
                or state["manifest"].get("full_source_sha256")!=file_sha256(trained)):
            raise ValueError("Require completed graph-head checkpoint from THIS full-text core")
        self.graph_head=UnifiedInstructionGraphBelief(hidden=128,layers=2).to(
            self.device).eval().requires_grad_(False)
        self.graph_head.load_state_dict(state["adapter"],strict=True)
        import spacy
        from scripts.analyze_instruction_landmark_association import PublicNames
        self.graph_nlp=spacy.load("en_core_web_sm",disable=["ner"])
        self.graph_names_class=PublicNames
        self.graph_source_record=None
        self.graph_proposal=empty_graph()
        self.graph_parser_status="no_episode"
        self.graph_last=None
        self.graph_checkpoint=Path(graph_head_checkpoint)

    def set_episode(self,episode,transform,catalog):
        super().set_episode(episode,transform,catalog)
        self.graph_source_record=attach_public_ids(
            self.record,episode,catalog,self.args.max_landmarks)
        self.graph_proposal=empty_graph()
        self.graph_last=None
        self.graph_parser_status="unresolved"
        from .structured_navigation import parse_instruction
        public=catalog.objects[episode.map_name]
        try:
            names=self.graph_names_class({episode.map_name:public})
            parsed=parse_instruction(episode.instruction,episode.map_name,
                                     self.graph_nlp,names)
            graph=compile_relation_graph(parsed,public,self.record["world_bounds"],
                                         world_north_verified=False)
            if len(graph["nodes"])>32 or len(graph["edges"])>64:
                raise ValueError("Graph exceeds capacity")
            self.graph_proposal=graph
            self.graph_parser_status="has_path" if graph["paths"] else "no_path"
        except (ValueError,KeyError,IndexError,TypeError):
            # No fake graph or GT fallback on unparsed public instructions.
            self.graph_proposal=empty_graph()
            self.graph_parser_status="parse_failed"

    @torch.no_grad()
    def predict(self,state,instruction):
        if self.record is None or instruction!=self.record["instruction"]:
            raise ValueError("Mismatched CityNav episode instruction")
        self._initialize_geometry_inputs()
        start=time.perf_counter()
        observed,reason=self._observed_inputs(state.current_observation)
        metadata=self._last_observation_metadata
        labels=(metadata["anchor_labels"] if metadata and observed is not None
                else None)
        with torch.autocast("cuda",enabled=False):
            result=self.unified.forward_full_relation(
                static_inputs=self.inputs,tokens=self.tokens.float(),
                token_mask=self.mask,node_emb=self.names.float(),
                observed=observed,acquisition=self.acquisition,
                anchor_labels=labels,relation_enabled=True)
            if self.graph_last is None:
                # Graph text nodes/edges are frozen BERT phrase embeddings.
                # Cache once per episode; never reparse every step.
                graph=graph_input_batch([self.graph_proposal],
                    [self.graph_source_record],self.bert,self.device,
                    entity_capacity=self.inputs["pair_geometry"].shape[2])
                self.graph_last=graph
            graph_out=self.graph_head(
                packed=self.graph_last["packed"],
                node_text=self.graph_last["node_text"],
                edge_text=self.graph_last["edge_text"],
                dense_context=result["field"].dense_context,
                pair_geometry=self.inputs["pair_geometry"],
                node_anchor=self.graph_last["node_anchor"],
                base_logits=result["field"].belief_logits,
            )
            field=replace_goal_field_logits(
                result["field"],self.inputs["grid_xy"],graph_out["logits"])
        audit=result["audit"]
        field,stable=self.stable_goal.select(
            field,self.inputs["grid_xy"],
            tracking=audit.get("acquisition"),
            visual_active=bool(audit["active"]))
        prediction=A3TrajectoryPolicy.predict_field(self,state,field)
        self.pgvb_calls+=int(audit["active"])
        self.pgvb_seconds+=time.perf_counter()-start
        prediction.diagnostics.update(
            relation_module="aerialvg_inspired_full+two_layer_graph",
            relation_control="full",visual_control="off",
            input_coverage="all_citynav_instructions",
            geometry_top1=int(result["geometry_field"].belief_logits[0].argmax()),
            relation_top1=int(result["relation_field"].belief_logits[0].argmax()),
            final_top1=int(field.belief_logits[0].argmax()),
            relation_delta_max=float(result["relation_delta"].abs().max()),
            graph_delta_max=float(graph_out["delta"].abs().max()),
            graph_supported=bool(graph_out["supported"][0]),
            graph_parser_status=self.graph_parser_status,
            graph_direct_paths=sum(p.get("hops")==1 for p in self.graph_proposal["paths"]),
            graph_indirect_paths=sum(p.get("hops",0)>1 for p in self.graph_proposal["paths"]),
            graph_public_anchor_count=int(self.graph_last["node_anchor"].sum()),
            rgb_active=bool(audit["active"]),
            rgb_reason=reason,
            stable_navigation_goal=stable,
            decision_seconds=time.perf_counter()-start,
        )
        if getattr(self,"full_relation_audit_callback",None) is not None:
            self.full_relation_audit_callback(state,result,field)
        return prediction
