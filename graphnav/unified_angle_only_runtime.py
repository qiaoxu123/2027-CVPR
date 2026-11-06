"""Opt-in FULL Val-Seen predicted-angle-only Unified V2 policy.

NO TRUE target bearing, target position or teacher labels are read here.
Public candidate bearing is computed before navigation solely from original
public source Landmark Polygon and fixed map grid. The new angle distribution
is predicted from the complete original instruction's frozen BERT embeddings.

Stop/trajectory/stabilizer unchanged. No distance/XY displacement is injected
into the new modality. The EXISTING Unified main model remains intact.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import torch

from .a3_runtime import A3TrajectoryPolicy
from .checkpointing import file_sha256
from .unified_angle_only_belief import (
    AngleOnlyBelief,public_candidate_bearings,named_anchor_mask,
)
from .unified_text_ablation_runtime import UnifiedTextAblationPolicy


def revised_field_with_angles(field,grid_xy,logits):
    """Update candidate IDs, XY and context coherently AFTER angle reranking."""
    if (logits.shape!=field.belief_logits.shape or
            field.dense_context is None or
            grid_xy.shape[:2]!=logits.shape):
        raise ValueError("Inconsistent GoalField/angular ranking inputs")
    k=field.candidate_ids.shape[-1]
    best=logits.topk(k,-1).indices
    newxy=grid_xy.gather(1,best[...,None].expand(-1,-1,2))
    dense=field.dense_context
    context=dense.gather(1,best[...,None].expand(-1,-1,dense.shape[-1]))
    return replace(field,belief_logits=logits,candidate_ids=best,
                   candidate_xy=newxy,candidate_context=context)


class UnifiedAngleOnlyPolicy(UnifiedTextAblationPolicy):
    def __init__(self,stage1,original_v2,trained,directory,*,
                 angle_head_checkpoint,public_objects,device="cuda",
                 relation_control="full",visual_control="off",
                 geometry_source="trained"):
        if (relation_control!="full" or visual_control!="off" or
                geometry_source!="trained"):
            raise ValueError("Angle experiment uses original full Geometry+Relation, RGB off")
        super().__init__(stage1,original_v2,trained,directory,
                         device=device,relation_control=relation_control,
                         visual_control=visual_control,
                         geometry_source=geometry_source,instruction_mode="full_text",public_objects=public_objects)
        state=torch.load(angle_head_checkpoint,map_location="cpu",weights_only=True)
        if (state.get("format")!="unified-v2-predicted-angle-only-v1" or
                state.get("epoch")!=state.get("manifest",{}).get("epochs") or
                state["manifest"].get("max_batches")!=0 or
                state["manifest"].get("baseline_sha256")!=file_sha256(trained)):
            raise ValueError("Require completed predicted-bearing checkpoint from original Unified V2")
        if state['manifest'].get('public_catalog_sha256')!=file_sha256(public_objects):raise ValueError('Public binding catalog changed')
        self.angle_head=AngleOnlyBelief(bins=8).to(self.device).eval().requires_grad_(False)
        self.angle_head.load_state_dict(state["head"],strict=True)
        self.angle_checkpoint=Path(angle_head_checkpoint)
        self.cached_candidate_angles=None
        self.cached_anchor_mask=None

    def set_episode(self,episode,transform,catalog):
        super().set_episode(episode,transform,catalog)
        self.cached_candidate_angles=None
        self.cached_anchor_mask=None

    @torch.no_grad()
    def predict(self,state,instruction):
        if self.record is None or instruction!=self.record["instruction"]:
            raise ValueError("Uninitialized or changed original CityNav instruction")
        self._initialize_geometry_inputs()
        if self.cached_candidate_angles is None:
            self.cached_candidate_angles=public_candidate_bearings(
                self.inputs["pair_geometry"])
            self.cached_anchor_mask=named_anchor_mask(
                self.inputs["entity_valid"],
                torch.tensor([len(self.record["landmarks"])],
                             device=self.device,dtype=torch.long))
        tic=time.perf_counter()
        # Original complete text and existing public Geometry/Relation stay in use.
        with torch.autocast("cuda",enabled=False):
            baseline=self.unified.forward_full_relation(
                static_inputs=self.inputs,
                tokens=self.tokens.float(),
                token_mask=self.mask,
                node_emb=self.names.float(),
                observed=None,acquisition=None,anchor_labels=None,
                relation_enabled=True)
            field=baseline["field"]
            augmented=self.angle_head(
                base_logits=field.belief_logits,
                candidate_bearings=self.cached_candidate_angles,
                landmark_valid=self.cached_anchor_mask,
                tokens=self.tokens.float(),
                token_mask=self.mask,
                node_emb=self.names.float(),
            )
            refined=revised_field_with_angles(
                field,self.inputs["grid_xy"],augmented["logits"])
        angular_field=refined
        refined,stable=self.stable_goal.select(
            refined,self.inputs["grid_xy"],tracking=None,visual_active=False)
        prediction=A3TrajectoryPolicy.predict_field(self,state,refined)
        prediction.diagnostics.update(
            modality="predicted_angular_direction_only",
            angle_GT_used_at_inference=False,
            angle_bins=8,
            angle_probabilities=augmented['angle_probs'][0].cpu().tolist(),
            named_anchor_mask=self.cached_anchor_mask[0].cpu().tolist(),
            rgb_active=False,
            relation_delta_max=float(baseline['relation_delta'].abs().max()),
            angle_supported=bool(augmented["supported"][0]),
            angle_delta_abs_max=float(augmented["delta"].abs().max()),
            angle_fusion_gain=float(self.angle_head.fusion_gain),
            baseline_top1=int(field.belief_logits[0].argmax()),
            refined_top1=int(refined.belief_logits[0].argmax()),
            original_full_text_preserved=True,
            stable_navigation_goal=stable,
            decision_seconds=time.perf_counter()-tic,
        )
        if getattr(self,"full_relation_audit_callback",None) is not None:
            self.full_relation_audit_callback(state,{**baseline,"field":angular_field,"angle_probs":augmented["angle_probs"],"angle_delta":augmented["delta"]},refined)
        return prediction
