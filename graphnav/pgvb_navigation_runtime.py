"""Opt-in PGVB-to-original-controller bridge for named-landmark diagnostics.

PGVB uses current real RGB, public named Polygon and camera footprint only.
The geometry/Stage1 baseline and trained trajectory/Stop head remain unchanged.
No private target, oracle location or future observation enters the policy.
"""
from dataclasses import replace
import time

import cv2
import numpy as np
import torch

from .appearance_matching import ObservedSigLIPMatchingEncoder
from .full_localization import GoalField
from .landmark_region_supervision import public_type
from .local_landmark_anchor import local_anchor_tokens, local_landmark_anchor, named_public_polygons
from .native_dense_regions import native_dense_regions, sparse_view
from .observed_patch_scales import masked_native_pool
from .polygon_guided_visual import PolygonGuidedVisualBelief
from .stage1 import preserved_proposal_order
from .stage1_runtime import GraphNavStage1Policy
from .world_likelihood_evidence import world_evidence_logits


def bounded_pgvb_field(field, grid_uv, local_uv, scores, valid, *, bound=0.5):
    """Conservative observed-only correction; preserve controller dense context.

    Unknown world cells receive exactly zero PGVB residual. A constant or absent
    visual score preserves the supplied field object and its candidate ordering.
    This is a fixed diagnostic fusion, not a separately trained fusion network.
    """
    if not (0.0 < bound <= 1.0):
        raise ValueError("bounded PGVB correction required")
    if scores.ndim != 2 or scores.shape != valid.shape or local_uv.shape != (*scores.shape, 2):
        raise ValueError("PGVB score/valid/UV mismatch")
    if not valid.any() or scores[valid].float().std(unbiased=False) < 1e-6:
        return field, dict(active=False, covered_cells=0, max_abs_delta=0.0)
    if field.dense_context is None:
        raise ValueError("Pretrained dense candidate context must be preserved")
    valid = valid.bool()
    q = field.belief_logits.shape[-1]
    zeros = torch.zeros((len(scores), q), device=scores.device, dtype=torch.float32)
    centered = scores.float().clone()
    for i in range(len(scores)):
        if valid[i].any():
            centered[i] -= centered[i, valid[i]].mean()
    baseline = zeros.log_softmax(-1)
    evidence = world_evidence_logits(zeros, local_uv, centered, valid) - baseline
    coverage = world_evidence_logits(zeros, local_uv, torch.ones_like(centered), valid) - baseline
    covered = coverage > 0.5
    delta = bound * torch.tanh(evidence / 5.0) * covered
    if not covered.any() or delta.abs().max() <= 1e-8:
        return field, dict(active=False, covered_cells=int(covered.sum()), max_abs_delta=0.0)
    logits = field.belief_logits.float() + delta
    ids = preserved_proposal_order(logits, field.candidate_ids)
    dense = field.dense_context
    coordinates = grid_uv.gather(1, ids[..., None].expand(-1, -1, 2))
    context = dense.gather(1, ids[..., None].expand(-1, -1, dense.shape[-1]))
    updated = GoalField(logits, ids, coordinates, context,
                        visual_coverage=covered.float(), dense_context=dense)
    return updated, dict(active=True, covered_cells=int(covered.sum()),
                         max_abs_delta=float(delta.abs().max().detach()),
                         changed_top1=bool((ids[:, 0] != field.candidate_ids[:, 0]).any()))


class PGVBNamedLandmarkPolicy(GraphNavStage1Policy):
    """Original GraphNav-V1 plus opt-in observed PGVB evidence on named landmarks."""

    def __init__(self, checkpoint, pgvb_checkpoint, directory, *, device="cuda", bound=0.5):
        super().__init__(checkpoint, directory, device=device)
        payload = torch.load(pgvb_checkpoint, map_location="cpu", weights_only=True)
        if "model" not in payload or "config" not in payload:
            raise ValueError("PGVB best.pt with model/config required")
        self.pgvb = PolygonGuidedVisualBelief().to(self.device).eval().requires_grad_(False)
        self.pgvb.load_state_dict(payload["model"], strict=True)
        if payload["config"].get("no_geometry_belief") is not True:
            raise ValueError("Expected geometry-independent PGVB checkpoint")
        self.enable_pgvb = False
        self.bound = float(bound)
        self.pgvb_calls = 0
        self._landmark = None
        self.audit_callback = None
        self.pgvb_seconds = 0.0

    def set_episode(self, episode, transform, catalog):
        super().set_episode(episode, transform, catalog)
        if len(episode.reference_names) != 1:
            raise ValueError("PGVB diagnostic expects one explicitly named goal landmark")
        name = episode.reference_names[0]
        matched = named_public_polygons(catalog.objects[episode.map_name], name)
        # CityReferCatalog uses int IDs; JSON audit catalogs use str IDs.
        objects = {str(oid): obj for oid, obj in catalog.objects[episode.map_name].items()}
        types = [objects[oid]["object_type"] for oid in matched["ids"]]
        typ = public_type(types)
        self._landmark = None
        if matched["contours"] and typ in ("Building", "Road"):
            self._landmark = dict(name=name, typ=typ, contours=matched["contours"])
        self.pgvb_calls = 0

    def _pgvb_forward(self, **kwargs):
        """Historical E4 scorer; unified policy overrides this scoring hook."""
        return self.pgvb(**kwargs)

    @torch.no_grad()
    def _observed_inputs(self, observation):
        landmark = self._landmark
        if landmark is None:
            return None, "unsupported_or_unresolved_name"
        anchor = local_landmark_anchor(landmark["contours"], observation)
        if not anchor["pixel_usable"] or anchor["positive_patches"] < 2:
            return None, "no_supported_local_anchor"
        dense = native_dense_regions(observation)
        sparse = sparse_view(dense)
        valid = np.asarray(sparse["valid"], bool) & (np.asarray(sparse["quality"])[:, 0] >= 0.9)
        world = np.asarray(sparse["world"], dtype=np.float32)
        bounds = np.asarray(self.record["world_bounds"], dtype=np.float32)
        local_uv = (world - bounds[:2]) / (bounds[2:] - bounds[:2])
        valid &= np.isfinite(local_uv).all(-1) & (local_uv >= 0).all(-1) & (local_uv <= 1).all(-1)
        if not valid.any():
            return None, "no_valid_visual_patches"

        bgr = np.asarray(observation["bgr"])
        batch = self.native.processor(
            images=[cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)],
            return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            patches = self.native.model.vision_model(**batch).last_hidden_state[0]
            masks = np.asarray(sparse["masks"], bool)
            flat = masks.reshape(-1, masks.shape[-1])
            nonempty = flat.any(-1)
            pooled = torch.zeros((len(flat), self.native.dimension), device=self.device)
            if nonempty.any():
                pooled[torch.from_numpy(nonempty).to(self.device)] = torch.nn.functional.normalize(
                    masked_native_pool(self.native.model.vision_model.head, patches,
                                       torch.from_numpy(flat[nonempty]).to(self.device)).float(), dim=-1)
            features = pooled.reshape(len(valid), 3, self.native.dimension)
            name_text = self.native.texts([landmark["name"]]).detach()
            type_text = self.native.texts([landmark["typ"]]).detach()
            polygon = local_anchor_tokens(landmark["contours"], observation["footprint_world"])
        return (dict(
            local_uv=torch.from_numpy(local_uv)[None].to(self.device),
            image_features=features[None],
            name_text=name_text,
            type_text=type_text,
            polygon_tokens=torch.from_numpy(polygon)[None].to(self.device),
            valid=torch.from_numpy(valid)[None].to(self.device),
        ), "available")

    @torch.no_grad()
    def _observed_scores(self, observation):
        """Historical PGVB scorer built from ONE shared observation extractor."""
        inputs, reason = self._observed_inputs(observation)
        if inputs is None:
            return None, reason
        model_inputs = {k: v for k, v in inputs.items() if k != "local_uv"}
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            scores = self._pgvb_forward(**model_inputs)
        return (inputs["local_uv"], scores.float(), inputs["valid"]), reason

    def _timed_scores(self, observation):
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        evidence, reason = self._observed_scores(observation)
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        self.pgvb_seconds = time.perf_counter() - start
        return evidence, reason

    @torch.no_grad()
    def predict(self, state, instruction):
        if not self.enable_pgvb:
            hook = None
            if self.audit_callback is not None:
                def capture_original(module, inputs, output):
                    self.audit_callback(output.belief, output.belief, None)
                hook = self.model.register_forward_hook(capture_original)
            try:
                prediction = super().predict(state, instruction)
            finally:
                if hook is not None: hook.remove()
            prediction.diagnostics.update(pgvb_enabled=False, pgvb_active=False, pgvb_seconds=0.0)
            return prediction

        evidence, reason = self._timed_scores(state.current_observation)
        result = dict(pgvb_enabled=True, pgvb_active=False, pgvb_reason=reason,
                      pgvb_delta_max_abs=0.0, pgvb_covered_cells=0)
        hook = None
        if evidence is not None:
            local_uv, scores, valid = evidence

            def apply_evidence(module, inputs, output):
                field, audit = bounded_pgvb_field(
                    output.belief, self.inputs["grid_xy"], local_uv, scores, valid, bound=self.bound)
                result.update(pgvb_active=audit["active"],
                              pgvb_delta_max_abs=audit["max_abs_delta"],
                              pgvb_covered_cells=audit["covered_cells"],
                              pgvb_changed_top1=audit.get("changed_top1", False))
                if self.audit_callback is not None:
                    self.audit_callback(output.belief, field, evidence)
                return replace(output, belief=field)

            hook = self.model.register_forward_hook(apply_evidence)
        elif self.audit_callback is not None:
            def capture_fallback(module, inputs, output):
                self.audit_callback(output.belief, output.belief, None)
            hook = self.model.register_forward_hook(capture_fallback)
        try:
            prediction = super().predict(state, instruction)
        finally:
            if hook is not None:
                hook.remove()
        self.pgvb_calls += int(result["pgvb_active"])
        prediction.diagnostics.update(result, pgvb_active_decisions_total=self.pgvb_calls, pgvb_seconds=self.pgvb_seconds)
        return prediction


class PGVBStandaloneNamedPolicy(PGVBNamedLandmarkPolicy):
    """PGVB conditional-visual Goal only; no A3 forward, no geometry logits.

    When the landmark has no usable local RGB observation, the policy explicitly
    abstains/holds position. No fabricated global Goal or geometry fallback.
    """

    def set_episode(self, episode, transform, catalog):
        super().set_episode(episode, transform, catalog)
        from .geometry import grid_centers
        self.bounds = torch.tensor(self.record["world_bounds"], device=self.device,
                                   dtype=torch.float32)[None]
        self.uv = torch.as_tensor(grid_centers(self.args.grid_size),
                                  device=self.device, dtype=torch.float32)[None]
        self.world = self.bounds[:, None, :2] + self.uv * (
            self.bounds[:, None, 2:] - self.bounds[:, None, :2])
        self.observed_steps = 0
        self.geometry_calls = 0

    @torch.no_grad()
    def predict(self, state, instruction):
        from types import SimpleNamespace
        from .a3_runtime import A3TrajectoryPolicy
        from .landmark_visual_only import visual_only_world_belief

        if instruction != self.record["instruction"]:
            raise ValueError("Changed instruction during PGVB-only episode")
        evidence, reason = self._timed_scores(state.current_observation)
        if evidence is None:
            if self.audit_callback is not None: self.audit_callback(None, None, None)
            # No search policy was trained. Explicit no-evidence abstention,
            # NOT a spurious guess at a citywide target location.
            return SimpleNamespace(
                xyz=np.asarray(state.pose)[:3].copy(), should_stop=False,
                diagnostics=dict(
                    policy="PGVB-only observed-domain conditional Goal",
                    selected_goal_xy=np.asarray(state.pose)[:2].tolist(),
                    pgvb_enabled=True, pgvb_active=False,
                    pgvb_reason=reason, pgvb_seconds=self.pgvb_seconds, geometry_branch_calls=0,
                    candidate_context="neutral_zero",
                    abstained_no_visual_goal=True,
                    observed_steps=self.observed_steps))
        local_uv, scores, valid = evidence
        result = visual_only_world_belief(
            local_uv=local_uv, scores=scores, valid=valid, grid_size=self.args.grid_size)
        if not bool(result["has_view"].all()):
            return SimpleNamespace(
                xyz=np.asarray(state.pose)[:3].copy(), should_stop=False,
                diagnostics=dict(
                    policy="PGVB-only observed-domain conditional Goal",
                    selected_goal_xy=np.asarray(state.pose)[:2].tolist(),
                    pgvb_enabled=True, pgvb_active=False,
                    pgvb_reason="no_world_grid_support", pgvb_seconds=self.pgvb_seconds,
                    geometry_branch_calls=0, candidate_context="neutral_zero",
                    abstained_no_visual_goal=True,
                    observed_steps=self.observed_steps))
        self.observed_steps += 1
        # Unobserved cells are 'unknown', not semantic negatives. The softmax
        # is explicitly conditioned on current observed support, so this
        # diagnostic does not claim to produce a global normalized belief.
        observed = result["covered"]
        logits = torch.where(observed, result["logits"],
                             torch.full_like(result["logits"], -100.))
        k = min(self.model.a3.top_k, logits.shape[-1])
        ids = logits.topk(k, dim=-1).indices
        dense = torch.zeros(1, logits.shape[-1], self.model.a3.hidden,
                            device=self.device, dtype=torch.float32)
        field = GoalField(
            belief_logits=logits,
            candidate_ids=ids,
            candidate_xy=self.uv.gather(1, ids[..., None].expand(-1, -1, 2)),
            candidate_context=dense.gather(
                1, ids[..., None].expand(-1, -1, dense.shape[-1])),
            dense_context=dense,
            visual_coverage=observed.float())
        if self.audit_callback is not None: self.audit_callback(None, field, evidence)
        prediction = A3TrajectoryPolicy.predict_field(self, state, field)
        prediction.diagnostics.update(
            policy="PGVB-only observed-domain conditional Goal",
            pgvb_enabled=True, pgvb_active=True,
            pgvb_reason="observed_conditioned_belief", pgvb_seconds=self.pgvb_seconds,
            geometry_branch_calls=0, candidate_context="neutral_zero",
            abstained_no_visual_goal=False,
            observed_steps=self.observed_steps,
            pgvb_covered_cells=int(observed.sum()),
            pgvb_top1_world_xy=self.world[0, ids[0, 0]].tolist())
        return prediction
