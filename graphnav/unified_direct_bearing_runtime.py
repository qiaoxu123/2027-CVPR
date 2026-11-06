"""Optional unified Full Relation policy for MAIN angular-consistency TRAIN.

This is NOT the PR25 angle-prediction auxiliary head. It loads a completed
fully-trained Geometry+Relation checkpoint with explicit cardinal predicate
IDs, and recomputes those IDs from public information BEFORE navigation.

No GT heading, target coordinates, distances or teacher labels at runtime.
The original main-model topology and controller/Stop remain unchanged.
"""
from __future__ import annotations

import torch

from .unified_text_ablation_runtime import UnifiedTextAblationPolicy
from .unified_direct_bearing_loss import predicate_ids_for_record


class UnifiedDirectBearingPolicy(UnifiedTextAblationPolicy):
    def __init__(self,stage1,original_v2,trained,directory,*,
                 instruction_mode="full_text",public_objects,
                 device="cuda",relation_control="full",
                 visual_control="off",geometry_source="trained"):
        if (instruction_mode!="full_text" or relation_control!="full"
                or visual_control!="off" or geometry_source!="trained"):
            raise ValueError("Direct angle-main requires complete original text, full relation, RGB off")
        super().__init__(stage1,original_v2,trained,directory,
                         instruction_mode=instruction_mode,public_objects=public_objects,
                         device=device,relation_control=relation_control,
                         visual_control=visual_control,geometry_source=geometry_source)
        bundle=torch.load(trained,map_location="cpu",weights_only=True)
        exp=bundle.get("angular_experiment")
        if not isinstance(exp,dict) or not exp.get("explicit_predicates"):
            raise ValueError("Angular main runtime requires a completed explicit-predicate checkpoint")
        self.angular_training_weight=float(exp["weight"])
        self._explicit_predicate_ids=None
        self.angular_reference_index=-1

    def set_episode(self,episode,transform,catalog):
        super().set_episode(episode,transform,catalog)
        capacity=self.args.max_landmarks+self.args.max_roads+self.args.max_relations
        ids,index=predicate_ids_for_record(
            self.record,self.public_objects[episode.map_name],capacity)
        self._explicit_predicate_ids=ids.to(self.device)[None]
        self.angular_reference_index=index
        # Main class has already reset the frozen Geometry cache. Keep that
        # and do not recompute any input from source target ID or GT.
        self.inputs=None

    def _initialize_geometry_inputs(self):
        super()._initialize_geometry_inputs()
        if self._explicit_predicate_ids is None:
            raise RuntimeError("Episode has no public predicate precomputation")
        if self.inputs["entity_valid"].shape!=self._explicit_predicate_ids.shape:
            raise ValueError("Model/public named-anchor capacity mismatch")
        self.inputs["anchor_predicate_ids"]=self._explicit_predicate_ids

    @torch.no_grad()
    def predict(self,state,instruction):
        result=super().predict(state,instruction)
        result.diagnostics.update(
            direct_world_cardinal_predicate=int(
                self._explicit_predicate_ids[0,0]) if self._explicit_predicate_ids is not None else 0,
            direct_world_cardinal_eligible=bool(self.angular_reference_index>=0),
            angular_consistency_training_weight=self.angular_training_weight,
            GT_world_bearing_at_inference=False,
        )
        return result
