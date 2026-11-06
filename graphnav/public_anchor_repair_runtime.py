"""Frozen Unified V2 PublicAnchor repair, only for previously UNBOUND episodes.

This is a paired *input-binding* intervention: same model checkpoint,
complete original text, map, initial pose, Stop and controller. Bound episodes
are bytewise-input-equivalent to the original Full Text baseline.

Not suitable for Test/Unseen performance reporting. Original Val-Seen has
already been inspected. This evaluates whether fixing public-map input
coverage changes outcomes for these known error cases.
"""
from __future__ import annotations

from pathlib import Path

from .unified_text_ablation_runtime import UnifiedTextAblationPolicy
from .public_anchor_repair import (
    repaired_public_text_record,sanitize_public_map
)


class UnifiedPublicAnchorRepairPolicy(UnifiedTextAblationPolicy):
    def __init__(self,*args,instruction_mode="full_text",public_objects,
                 repair_mode="both",**kwargs):
        if instruction_mode!="full_text" or repair_mode not in ("geometry","capacity","both"):
            raise ValueError("Intervention requires original full instruction and explicit repair mode")
        if kwargs.get("visual_control","off")!="off":
            raise ValueError("Strict paired baseline requires RGB off")
        super().__init__(*args,instruction_mode=instruction_mode,
                         public_objects=public_objects,**kwargs)
        self.repair_mode=repair_mode
        self._safe_maps={}
        self._safe_ids={}
        self.repair_audit=None

    def _catalog_view(self,map_name):
        if self.repair_mode=="capacity":
            return self.public_objects,()
        if map_name not in self._safe_maps:
            updated,updated_ids=sanitize_public_map(self.public_objects[map_name])
            self._safe_maps[map_name]=updated
            self._safe_ids[map_name]=tuple(updated_ids)
        catalog=dict(self.public_objects)
        catalog[map_name]=self._safe_maps[map_name]
        return catalog,self._safe_ids[map_name]

    def set_episode(self,episode,transform,catalog):
        # The parent computes the EXACT frozen PR23 full_text baseline input.
        # Target ID/GT and trajectories are not passed into this method.
        super().set_episode(episode,transform,catalog)
        base=self.record
        before_ids=tuple(str(x["id"]) for x in base["landmarks"])
        if before_ids:
            self.repair_audit=dict(repair_mode=self.repair_mode,
                original_bound=len(before_ids),repaired_bound=len(before_ids),
                before_ids=list(before_ids),after_ids=list(before_ids),
                applied=False,reason="original_binding_already_nonempty",
                GT_used=False)
            return
        bounds=[transform.x_min,transform.y_min,transform.x_max,transform.y_max]
        view,ids=self._catalog_view(episode.map_name)
        record,audit=repaired_public_text_record(
            episode.map_name,episode.instruction,bounds,view,
            self.public_name_matcher,mode="full_text",
            max_landmarks=self.args.max_landmarks,
            repair_geometry=False,
            repair_overflow=self.repair_mode in ("capacity","both"),
            preverified_repaired_ids=ids)
        after_ids=tuple(str(x["id"]) for x in record["landmarks"])
        self.repair_audit=dict(audit,repair_mode=self.repair_mode,
            original_bound=0,repaired_bound=len(after_ids),
            before_ids=[],after_ids=list(after_ids),
            applied=bool(after_ids),
            reason="recovered_from_public_catalog" if after_ids
                   else "unresolved_without_safe_public_polygon",
            GT_used=False)
        if after_ids:
            self.record=record
            self.inputs=None
            self.unified.reset_episode()
        self.relation_coverage={
            **(self.relation_coverage or {}),
            "binding_repair_applied":bool(after_ids),
            "binding_repair_mode":self.repair_mode,
            "binding_repair_before":0,
            "binding_repair_after":len(after_ids),
        }

    def predict(self,state,instruction):
        result=super().predict(state,instruction)
        audit=self.repair_audit
        if audit is None:raise RuntimeError("Binding repair audit missing")
        result.diagnostics.update(
            binding_repair_mode=self.repair_mode,
            binding_repair_applied=bool(audit["applied"]),
            binding_repair_before=int(audit["original_bound"]),
            binding_repair_after=int(audit["repaired_bound"]),
            public_binding_only=True,
            gt_selected_polygon=False,
        )
        return result
