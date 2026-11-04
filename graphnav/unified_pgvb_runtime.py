"""Single observed PGVB visual path with frozen A3 geometry and controller.

Unlike historical E4, this NEVER executes GraphNavStage1.forward(), its
MultiScaleAppearanceMatcher, RGB residual or visual-history branch. The sole
per-frame image encoding is the SigLIP2 call in PGVB's observed patch scorer.
Historical checkpoints remain readable for A3 and Trajectory/Stop.
"""
import json

import torch

from .a3_runtime import A3TrajectoryPolicy
from .candidate_geometry import FullCandidateDataset, collate_full
from .full_experiment import _batch_inputs
from .pgvb_navigation_runtime import (
    PGVBNamedLandmarkPolicy, bounded_pgvb_field,
)
from .stage1_runtime import GraphNavStage1Policy
from .unified_pgvb import UnifiedPGVB
from .unified_instruction import parse_unified_instruction
from .landmark_alias import canonical_name


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
        # Strictly load the historical checkpoint first, then remove its
        # dormant legacy RGB heads from THIS new policy instance. The source
        # checkpoint and historical E2/E4 runners remain untouched.
        del self.model.matcher
        del self.model.history
        del self.model.visual_gain
        del self.model.a3.vision_proj
        del self.model.a3.visual_score
        del self.visual
        self.use_history = False
        bundle = torch.load(
            unified_checkpoint, map_location="cpu", weights_only=True,
        )
        if bundle.get("format") != "unified-pgvb-v1":
            raise ValueError("Expected unified-pgvb-v1 checkpoint")
        config = bundle["config"]
        if config.get("target_ready"):
            raise ValueError("Named-landmark pilot cannot validate target head")
        self.unified = UnifiedPGVB().to(self.device)
        self.unified.load_state_dict(bundle["model"], strict=True)
        self.unified.direction_ready = bool(config.get("direction_ready", False))
        self.unified.target_ready = False
        self.unified.eval().requires_grad_(False)
        # Historical observation routine calls precisely this shared visual
        # scorer. No additional copy of PGVB or SigLIP2 is allocated.
        self.pgvb = self.unified.pgvb
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
        self.inputs, tokens, mask, names = _batch_inputs(
            collate_full([ds[0]]), self.bert, self.device, "geometry",
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            self.base = self.model.a3(
                self.inputs, variant="geometry",
                tokens=tokens, token_mask=mask, node_emb=names,
            )
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
        evidence, reason = self._timed_scores(state.current_observation)
        field = self.base
        audit = dict(
            active=False, covered_cells=0, max_abs_delta=0.,
            changed_top1=False,
        )
        if evidence is not None:
            local_uv, scores, valid = evidence
            field, audit = bounded_pgvb_field(
                self.base, self.inputs["grid_xy"],
                local_uv, scores, valid, bound=self.bound,
            )
        if self.audit_callback is not None:
            self.audit_callback(self.base, field, evidence)
        prediction = A3TrajectoryPolicy.predict_field(self, state, field)
        self.pgvb_calls += int(audit["active"])
        language = getattr(self, "instruction_spec", {})
        prediction.diagnostics.update(
            policy="UnifiedPGVB single visual path + frozen A3/control",
            pgvb_enabled=True,
            pgvb_active=audit["active"],
            pgvb_reason=reason,
            pgvb_seconds=self.pgvb_seconds,
            pgvb_delta_max_abs=audit["max_abs_delta"],
            pgvb_covered_cells=audit["covered_cells"],
            pgvb_changed_top1=audit["changed_top1"],
            pgvb_active_decisions_total=self.pgvb_calls,
            old_stage1_rgb_executed=False,
            siglip2_rgb_forward=(evidence is not None),
            direction_head_ready=self.unified.direction_ready,
            target_head_ready=False,
            language_parser=language.get("version", "not_initialized"),
            language_intent=language.get("intent", "unknown"),
            named_public_parts=len(
                language.get("landmark", {}).get("public_ids", [])
            ) if language.get("landmark") else 0,
            language_target_gt_used=False,
        )
        return prediction
