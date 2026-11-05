"""Full CityNav relation-aware UnifiedPGVB inference, with frozen original Stop.

Every CityNav instruction is accepted, including indirect relations,
unmatched place names and tasks without a public anchor. No GT data enters
model construction, set_episode or predict.
"""
from pathlib import Path
import time

import torch

from .a3_runtime import A3TrajectoryPolicy, public_static_record
from .checkpointing import file_sha256
from .full_relation_training import load_full_relation_model
from .landmark_region_supervision import public_type
from .local_landmark_anchor import named_public_polygons
from .stable_navigation_goal import StableNavigationGoal
from .unified_pgvb_runtime import UnifiedPGVBNamedLandmarkPolicy


class UnifiedPGVBFullRelationPolicy(UnifiedPGVBNamedLandmarkPolicy):
    def __init__(self, stage1, original_v2, trained, directory, *, device="cuda",
                 relation_control="full", visual_control="tracked"):
        if relation_control not in ("full", "off"):
            raise ValueError("Invalid learned relation ablation")
        if visual_control not in ("tracked", "off"):
            raise ValueError("Invalid PGVB visual ablation")
        super().__init__(stage1, original_v2, directory, device=device,
                         mode="tracked-stable" if visual_control == "tracked" else "geometry")
        self.relation_control = relation_control
        self.visual_control = visual_control
        state = torch.load(trained, map_location="cpu", weights_only=True)
        if state.get("format") != "unified-pgvb-full-relation-v1":
            raise ValueError("A trained full-instruction checkpoint is required")
        sources = state.get("source_sha256", {})
        if (sources.get("original_v2") != file_sha256(original_v2)
                or sources.get("stage1") != file_sha256(stage1)):
            raise ValueError("Relation checkpoint/source controller mismatch")
        if state.get("completed_epochs", 0) < 1:
            raise ValueError("Incomplete full relation training")
        self.unified = load_full_relation_model(state, self.device).eval().requires_grad_(False)
        self.unified.full_relation_ready = True
        self.full_relation_checkpoint = Path(trained)
        self.relation_coverage = None

    def set_episode(self, episode, transform, catalog):
        if not episode.instruction:
            raise ValueError("Empty CityNav instruction")
        # Use the ORIGINAL source dataset's public referenced names, already
        # supplied by CityReferCatalog. Do NOT run the 573-query strict parser.
        self.record = public_static_record(
            episode, transform, catalog,
            max_landmarks=self.args.max_landmarks,
        )
        public_objects = catalog.objects[episode.map_name]
        names = tuple(episode.reference_names or
                      tuple(getattr(ref, "name", "") for ref in episode.references))
        names = tuple(x for x in names if x)
        self._landmark = None
        # Visual anchor is only permitted for a uniquely bound public name;
        # geometry/relation still run for all ambiguous or ungrounded queries.
        if len(names) == 1:
            try:
                found = named_public_polygons(public_objects, names[0])
                objects = {str(k): v for k, v in public_objects.items()}
                kind = public_type(
                    [objects[str(k)]["object_type"] for k in found["ids"]]
                )
                if len(found["ids"]) == 1 and kind in ("Building", "Road"):
                    self._landmark = dict(
                        name=names[0], typ=kind, contours=found["contours"]
                    )
            except (KeyError, ValueError):
                self._landmark = None
        self._public_region = None
        self._neighbor_geometry = []
        self.instruction_spec = None
        self.inputs = None
        self.acquisition.reset()
        self.stable_goal.reset()
        self.pgvb_calls = 0
        self.pgvb_seconds = 0.0
        self._last_observation_metadata = None
        self.unified.reset_episode()
        self.relation_coverage = dict(
            public_reference_names=len(names),
            visual_anchor_available=self._landmark is not None,
        )

    @torch.no_grad()
    def predict(self, state, instruction):
        if self.record is None or instruction != self.record["instruction"]:
            raise ValueError("Uninitialized or mismatched CityNav instruction")
        self._initialize_geometry_inputs()
        start = time.perf_counter()
        observed, reason = self._observed_inputs(state.current_observation)
        metadata = self._last_observation_metadata
        labels = metadata["anchor_labels"] if metadata and observed is not None else None
        # The small Belief core must be FP32, as established by PR #16.
        with torch.autocast("cuda", enabled=False):
            result = self.unified.forward_full_relation(
                static_inputs=self.inputs, tokens=self.tokens.float(),
                token_mask=self.mask, node_emb=self.names.float(),
                observed=observed, acquisition=self.acquisition,
                anchor_labels=labels,
                relation_enabled=self.relation_control == "full",
            )
        audit = result["audit"]
        field, stable = self.stable_goal.select(
            result["field"], self.inputs["grid_xy"],
            tracking=audit.get("acquisition"),
            visual_active=bool(audit["active"]),
        )
        prediction = A3TrajectoryPolicy.predict_field(self, state, field)
        self.pgvb_calls += int(audit["active"])
        self.pgvb_seconds += time.perf_counter() - start
        prediction.diagnostics.update(
            relation_module="aerialvg_inspired_full",
            relation_control=self.relation_control,
            visual_control=self.visual_control,
            input_coverage="all_citynav_instructions",
            geometry_top1=int(result["geometry_field"].belief_logits[0].argmax()),
            relation_top1=int(result["relation_field"].belief_logits[0].argmax()),
            final_top1=int(field.belief_logits[0].argmax()),
            relation_delta_max=float(result["relation_delta"].abs().max()),
            rgb_active=bool(audit["active"]),
            rgb_reason=reason,
            anchor_audit=self.relation_coverage,
            stable_navigation_goal=stable,
            decision_seconds=time.perf_counter()-start,
        )
        return prediction
