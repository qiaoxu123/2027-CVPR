"""Full-text Unified V2 with matching public-only grouped Predicate inputs.

A complete independently PUBLIC-BLIND trained checkpoint is required.
GT target identity, target_xy and ground truth yaw do not participate in
binding or inference. Stop, controller, thresholds and original RGB state
are inherited unchanged from existing UnifiedTextAblationPolicy.

The optional same-name group cardinal LOSS trains only; evaluation supplies
the text-derived predicate IDs for each individual public Polygon instance.
"""
from __future__ import annotations

import torch

from .unified_text_ablation_runtime import UnifiedTextAblationPolicy
from .unified_polygon_group import public_cardinal_group
from .checkpointing import file_sha256


class UnifiedPublicPolygonGroupPolicy(UnifiedTextAblationPolicy):
    def __init__(self,stage1,original_v2,trained,directory,*,
                 instruction_mode="full_text",public_objects,
                 device="cuda",relation_control="full",
                 visual_control="off",geometry_source="trained"):
        if (instruction_mode!="full_text" or relation_control!="full"
                or visual_control!="off" or geometry_source!="trained"):
            raise ValueError("Grouped-public comparison uses full text, full relation, RGB off")
        super().__init__(stage1,original_v2,trained,directory,
                         instruction_mode=instruction_mode,public_objects=public_objects,
                         device=device,relation_control=relation_control,
                         visual_control=visual_control,geometry_source=geometry_source)
        payload=torch.load(trained,map_location="cpu",weights_only=True)
        experiment=payload.get("angular_experiment")
        if (payload.get("smoke") or
                payload.get("completed_epochs",0)<1 or
                payload.get("completed_epochs")!=payload.get("planned_epochs") or
                not isinstance(experiment,dict) or
                experiment.get("protocol")!="same-name-public-polygon-group-v1"):
            raise ValueError("Requires a COMPLETE main polygon-group training checkpoint")
        if payload.get("source_sha256",{}).get("public_objects")!=file_sha256(public_objects):
            raise ValueError("Runtime PUBLIC polygon catalog differs from TRAIN")
        if experiment.get("public_binding_algorithm")!="instruction_public_blind_PR23":
            raise ValueError("Only fully public-blind matched checkpoints supported")
        self.group_predicates_enabled=experiment["predicate_mode"]=="group"
        self.group_angular_mode=experiment["angular_loss_mode"]
        self._group_predicate_ids=None
        self._group_member_mask=None
        self._group_status="uninitialized"

    def set_episode(self,episode,transform,catalog):
        super().set_episode(episode,transform,catalog)
        cap=self.args.max_landmarks+self.args.max_roads+self.args.max_relations
        ids,mask,status=public_cardinal_group(
            self.record,self.public_name_matcher,self.public_objects,cap,
            enabled=self.group_predicates_enabled)
        self._group_predicate_ids=ids[None].to(self.device)
        self._group_member_mask=mask
        self._group_status=status
        self.inputs=None

    def _initialize_geometry_inputs(self):
        super()._initialize_geometry_inputs()
        if self._group_predicate_ids is None:
            raise RuntimeError("Episode group predicates uninitialized")
        if self.inputs["entity_valid"].shape!=self._group_predicate_ids.shape:
            raise ValueError("PUBLIC landmark group/entity node dimension mismatch")
        self.inputs["anchor_predicate_ids"]=self._group_predicate_ids

    @torch.no_grad()
    def predict(self,state,instruction):
        output=super().predict(state,instruction)
        output.diagnostics.update(
            polygon_group_status=self._group_status,
            polygon_group_size=int(self._group_member_mask.sum()),
            polygon_group_predicate_enabled=self.group_predicates_enabled,
            polygon_group_angular_loss=self.group_angular_mode,
            public_name_only_bind=True,
            target_GT_used_to_select_polygon=False,
        )
        return output
