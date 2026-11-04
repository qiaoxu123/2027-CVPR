"""UnifiedPGVB-v2 runtime: ONE geometry+vision GoalField model, ONE controller.

No subclass of historical GraphNavStage1Policy/PGVBNamedLandmarkPolicy.
No standalone A3 model and no legacy RGB Matcher/History/Residual.
The original Stage1 file supplies controller weights and a compatibility SHA;
all geometry weights live only in the unified-pgvb-v2 checkpoint.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from .a3_runtime import A3TrajectoryPolicy, public_static_record
from .appearance_matching import ObservedSigLIPMatchingEncoder
from .candidate_geometry import FullCandidateDataset, collate_full
from .checkpointing import file_sha256
from .full_experiment import _batch_inputs, new_model
from .landmark_alias import canonical_name
from .landmark_region_supervision import public_type
from .local_landmark_anchor import named_public_polygons
from .pgvb_observation import encode_landmark_observation
from .token_language import FrozenBERTTokens
from .trajectory_policy import CandidateTrajectoryStop
from .unified_instruction import parse_unified_instruction
from .unified_pgvb import UnifiedPGVB


class UnifiedPGVBNamedLandmarkPolicy:
    """Single model generates Belief and Top-K; frozen controller flies it."""

    def __init__(
        self, checkpoint, unified_checkpoint, directory,
        *, device="cuda", bound=0.5,
    ):
        self.device = torch.device(device)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        stage1 = torch.load(
            checkpoint, map_location="cpu", weights_only=True,
        )
        protocol = stage1.get("protocol", {})
        if (
            protocol.get("name") != "graphnav-v1-stage1"
            or protocol.get("active_observation")
        ):
            raise ValueError("Expected non-active frozen GraphNav Stage1 protocol")
        bundle = torch.load(
            unified_checkpoint, map_location="cpu", weights_only=True,
        )
        if bundle.get("format") != "unified-pgvb-v2":
            raise ValueError("Expected integrated-geometry unified-pgvb-v2")
        config = bundle["config"]
        if config.get("source", {}).get("stage1_sha256") != file_sha256(checkpoint):
            raise ValueError("Integrated geometry and controller source mismatch")
        if (
            not config.get("geometry_integrated")
            or not config.get("single_rgb_encoder")
        ):
            raise ValueError("Checkpoint must contain one full geometry+visual model")
        if (config.get("target_ready") or config.get("jointly_trained")
                or config.get("joint_fusion_ready")):
            raise ValueError("Unvalidated target/joint model cannot use landmark pilot")

        self.args = SimpleNamespace(**stage1["a3_args"])
        if (
            int(self.args.grid_size) != 28
            or int(self.args.hidden) != 128
        ):
            raise ValueError("Controller protocol requires 28x28/128-d GoalField")
        geometry = new_model(self.args, self.device)
        del geometry.vision_proj
        del geometry.visual_score
        self.unified = UnifiedPGVB(geometry=geometry).to(self.device)
        self.unified.load_state_dict(bundle["model"], strict=True)
        self.unified.direction_ready = bool(config.get("direction_ready", False))
        self.unified.target_ready = False
        self.unified.joint_ready = False
        self.unified.eval().requires_grad_(False)

        # The controller is the only remaining independent trained module.
        weights = {
            name[len("trajectory."):]: value
            for name, value in stage1["model"].items()
            if name.startswith("trajectory.")
        }
        self.head = CandidateTrajectoryStop(
            hidden=self.args.hidden,
        ).to(self.device).eval().requires_grad_(False)
        self.head.load_state_dict(weights, strict=True)
        self.stop_threshold = .3
        self.stop_positive_weight = 20.
        self.execution = "guarded-blend"
        self.explore_candidates = 0

        self.bert = FrozenBERTTokens(
            self.args.text_backbone,
            device=device,
            max_length=getattr(self.args, "max_text_length", 96),
        )
        expected_bert = stage1.get("config", {}).get(
            "frozen_backbones", {},
        ).get("bert_revision")
        if expected_bert and self.bert.revision != expected_bert:
            raise ValueError("Original BERT frozen revision mismatch")
        self.native = ObservedSigLIPMatchingEncoder(
            protocol["backbone"], device=device, microbatch=1,
        )
        if (
            getattr(self.native.model.config, "_commit_hash", None)
            != protocol["backbone_revision"]
        ):
            raise ValueError("Original SigLIP2 frozen revision mismatch")

        self.bound = float(bound)
        if not 0 < self.bound <= 1:
            raise ValueError("PGVB evidence bound must be in (0, 1]")
        self.record = None
        self.inputs = None
        self.instruction_spec = None
        self._landmark = None
        self.audit_callback = None
        self.pgvb_seconds = 0.
        self.pgvb_calls = 0
        self.visited_goal_cells = set()

    def set_episode(self, episode, transform, catalog):
        """Resolve public named Polygon BEFORE any network inference."""
        public_objects = catalog.objects[episode.map_name]
        spec = parse_unified_instruction(
            episode.instruction, episode.map_name, public_objects,
        )
        landmark = spec.get("landmark")
        references = tuple(episode.reference_names)
        if (
            not spec["approved_for_inference"]
            or spec["intent"] != "navigate_to_landmark"
            or landmark is None
        ):
            raise ValueError(
                "UnifiedPGVB named-goal pilot refuses non-landmark instruction: "
                + spec["reason"]
            )
        if (
            len(references) != 1
            or canonical_name(references[0])
            != canonical_name(landmark["name"])
        ):
            raise ValueError("The named goal and supplied public reference differ")
        matched = named_public_polygons(public_objects, landmark["name"])
        objects = {str(oid): obj for oid, obj in public_objects.items()}
        kinds = [objects[obj_id]["object_type"] for obj_id in matched["ids"]]
        typ = public_type(kinds)
        if not matched["contours"]:
            raise ValueError("No public Polygon for named goal")
        # All named tasks stay in the evaluation denominator. A mixed-type
        # public name is an ambiguous GEOMETRY-only goal (the legacy public
        # union protocol); PGVB does not pretend to visually identify it.
        self._landmark = (
            dict(name=landmark["name"], typ=typ, contours=matched["contours"])
            if typ in ("Building", "Road") and landmark["visual_supported"]
            else None
        )
        self.record = public_static_record(
            episode, transform, catalog,
            max_landmarks=self.args.max_landmarks,
        )
        self.instruction_spec = spec
        self.inputs = None
        self.visited_goal_cells = set()
        self.pgvb_calls = 0
        self.unified.reset_episode()

    def _observed_inputs(self, observation):
        return encode_landmark_observation(
            observation, self._landmark, self.record,
            self.native, self.device,
        )

    @torch.no_grad()
    def _initialize_geometry_inputs(self):
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
        self.bounds = torch.tensor(
            self.record["world_bounds"], device=self.device,
            dtype=torch.float32,
        )[None]
        self.world = (
            self.bounds[:, None, :2] + self.inputs["grid_xy"]
            * (self.bounds[:, None, 2:] - self.bounds[:, None, :2])
        )

    @torch.no_grad()
    def predict(self, state, instruction):
        if self.record is None or instruction != self.record["instruction"]:
            raise ValueError("Instruction changed or episode not initialized")
        self._initialize_geometry_inputs()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
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
        self.pgvb_seconds = time.perf_counter() - start
        base, field = result["geometry_field"], result["field"]
        evidence, audit = result["evidence"], result["audit"]
        if self.audit_callback is not None:
            self.audit_callback(base, field, evidence)
        prediction = A3TrajectoryPolicy.predict_field(self, state, field)
        self.pgvb_calls += int(audit["active"])
        spec = self.instruction_spec
        prediction.diagnostics.update(
            policy="UnifiedPGVB-v2 owned geometry+single RGB; frozen controller",
            pgvb_enabled=True, pgvb_active=audit["active"],
            pgvb_reason=reason, pgvb_seconds=self.pgvb_seconds,
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
            language_parser=spec["version"], language_intent=spec["intent"],
            named_public_parts=len(
                spec["landmark"].get("public_ids", []),
            ),
            language_target_gt_used=False,
            landmark_visual_supported=self._landmark is not None,
        )
        return prediction
