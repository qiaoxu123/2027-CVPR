"""UnifiedPGVB owns geometry and the sole RGB visual branch.

The old GraphNavStage1 shell exists ONLY for strict legacy weight loading, then
is removed. No external A3.forward, Matcher, visual residual or visual-history
network remains in the live policy. Only the frozen Trajectory/Stop head is
retained separately as the controller.
"""
import json
import time

import torch

from .a3_runtime import A3TrajectoryPolicy
from .candidate_geometry import FullCandidateDataset, collate_full
from .full_experiment import _batch_inputs
from .pgvb_navigation_runtime import PGVBNamedLandmarkPolicy
from .stage1_runtime import GraphNavStage1Policy
from .unified_pgvb import UnifiedPGVB
from .unified_instruction import parse_unified_instruction
from .landmark_alias import canonical_name
from .checkpointing import file_sha256


class UnifiedPGVBNamedLandmarkPolicy(PGVBNamedLandmarkPolicy):
    """A3 geometric Goal + ONE PGVB RGB update -> frozen navigation controller."""

    def __init__(
        self, checkpoint, unified_checkpoint, directory,
        *, device="cuda", bound=0.5,
    ):
        # Only use the legacy checkpoint's geometry and controller weights.
        # Its visual matcher/history/residual are NEVER called by predict().
        GraphNavStage1Policy.__init__(
            self, checkpoint, directory, device=device,
        )
        bundle = torch.load(
            unified_checkpoint, map_location="cpu", weights_only=True,
        )
        if bundle.get("format") != "unified-pgvb-v2":
            raise ValueError("Expected integrated-geometry unified-pgvb-v2 checkpoint")
        config = bundle["config"]
        if config.get("source", {}).get("stage1_sha256") != file_sha256(checkpoint):
            raise ValueError("Integrated geometry and controller checkpoint mismatch")
        if config.get("target_ready"):
            raise ValueError("Named-landmark pilot cannot validate target head")
        if config.get("jointly_trained"):
            raise ValueError("Unverified jointly-trained model must not use pilot loader")
        geometry = self.model.a3
        # Legacy A3 visual layers are inactive in geometry mode; remove them
        # from the *internal* unified geometry model to avoid a second RGB head.
        del geometry.vision_proj
        del geometry.visual_score
        self.unified = UnifiedPGVB(geometry=geometry).to(self.device)
        self.unified.load_state_dict(bundle["model"], strict=True)
        self.unified.direction_ready = bool(config.get("direction_ready", False))
        self.unified.target_ready = False
        self.unified.eval().requires_grad_(False)
        # Retain ONLY the pretrained controller, never a second localizer.
        self.head = self.model.trajectory
        self.head.eval().requires_grad_(False)
        del self.model
        del self.visual
        self.use_history = False
        self.bound = float(bound)
        if not 0 < self.bound <= 1:
            raise ValueError("PGVB correction bound must be in (0,1]")
        self.enable_pgvb = True
        self.pgvb_calls = 0
        self.pgvb_seconds = 0.0
        self.audit_callback = None
        self._landmark = None

    def set_episode(self, episode, transform, catalog):
        """Reject target-vs-reference confusion BEFORE constructing geometry."""
        public_objects = catalog.objects[episode.map_name]
        spec = parse_unified_instruction(
            episode.instruction, episode.map_name, public_objects,
        )
        named = spec.get("landmark")
        references = tuple(episode.reference_names)
        if (not spec["approved_for_inference"]
                or spec["intent"] != "navigate_to_landmark"
                or named is None):
            raise ValueError(
                "UnifiedPGVB named-goal pilot refuses non-landmark instruction: "
                + spec["reason"]
            )
        if (len(references) != 1
                or canonical_name(references[0]) != canonical_name(named["name"])):
            raise ValueError("The named goal and supplied public reference differ")
        super().set_episode(episode, transform, catalog)
        if self._landmark is None:
            raise ValueError("No valid public Polygon for named goal")
        self.instruction_spec = spec
        self.unified.reset_episode()

    def _pgvb_forward(self, **kwargs):
        """One shared PGVB vision network, driven by parsed instruction."""
        return self.unified.forward_structured(
            self.instruction_spec, **kwargs
        )["landmark_scores"]

    @torch.no_grad()
    def _initialize_geometry(self):
        if self.inputs is not None:
            return
        path = self.directory / "current_public_query.jsonl"
        path.write_text(json.dumps(self.record) + "\n")
        ds = FullCandidateDataset(
            path, grid_size=self.args.grid_size,
            max_landmarks=self.args.max_landmarks,
            max_roads=self.args.max_roads,
            max_relations=self.args.max_relations,
            prior_sigma_m=self.args.prior_sigma_m,
            rule_sigma_m=self.args.rule_sigma_m,
        )
        self.inputs, self.tokens, self.mask, self.names = _batch_inputs(
            collate_full([ds[0]]), self.bert, self.device, "geometry",
        )
        # No independent geometry forward. UnifiedPGVB owns the A3 weights
        # and computes/caches the geometry field within forward_field().
        self.bounds = torch.tensor(
            self.record["world_bounds"],
            device=self.device, dtype=torch.float32,
        )[None]
        self.world = (
            self.bounds[:, None, :2]
            + self.inputs["grid_xy"] *
            (self.bounds[:, None, 2:] - self.bounds[:, None, :2])
        )

    @torch.no_grad()
    def predict(self, state, instruction):
        if instruction != self.record["instruction"]:
            raise ValueError("Instruction changed within episode")
        self._initialize_geometry()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        observed, reason = self._observed_inputs(state.current_observation)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            result = self.unified.forward_structured(
                self.instruction_spec,
                static_inputs=self.inputs,
                tokens=self.tokens,
                token_mask=self.mask,
                node_emb=self.names,
                observed=observed,
                bound=self.bound,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.pgvb_seconds = time.perf_counter() - started
        base, field = result["geometry_field"], result["field"]
        evidence, audit = result["evidence"], result["audit"]
        if self.audit_callback is not None:
            self.audit_callback(base, field, evidence)
        prediction = A3TrajectoryPolicy.predict_field(self, state, field)
        self.pgvb_calls += int(audit["active"])
        language = self.instruction_spec
        prediction.diagnostics.update(
            policy="UnifiedPGVB-v2 integrated geometry+single RGB; frozen controller",
            pgvb_enabled=True,
            pgvb_active=audit["active"],
            pgvb_reason=reason,
            pgvb_seconds=self.pgvb_seconds,
            pgvb_delta_max_abs=audit["max_abs_delta"],
            pgvb_covered_cells=audit["covered_cells"],
            pgvb_changed_top1=audit["changed_top1"],
            pgvb_active_decisions_total=self.pgvb_calls,
            old_stage1_rgb_executed=False,
            siglip2_rgb_forward=(observed is not None),
            geometry_owner="UnifiedPGVB.geometry",
            standalone_a3_forward=False,
            direction_head_ready=self.unified.direction_ready,
            target_head_ready=False,
            language_parser=language["version"],
            language_intent=language["intent"],
            named_public_parts=len(
                language.get("landmark", {}).get("public_ids", [])
            ) if language.get("landmark") else 0,
            language_target_gt_used=False,
        )
        return prediction
