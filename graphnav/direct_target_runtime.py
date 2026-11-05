"""Trained direct-target adapter; unchanged StableGoal and Trajectory/Stop."""
import json
import time
from pathlib import Path
import torch
from .unified_pgvb_runtime import UnifiedPGVBNamedLandmarkPolicy
from .direct_target_data import load_model, relation_geometry
from .a3_runtime import public_static_record, A3TrajectoryPolicy
from .pgvb_observation import encode_landmark_observation
from .local_landmark_anchor import named_public_polygons
from .checkpointing import file_sha256


def admissible_target_rgb(tracking, metadata, observed):
    """Provisional anchor evidence gate, NOT a target-instance identity proof."""
    if observed is None or metadata is None:
        return False, "no_supported_anchor_rgb"
    if metadata.get("registration_only") is not True:
        raise ValueError("Unexpected landmark evidence provenance")
    if tracking is None or tracking.get("state") != "tracking":
        return False, "anchor_tracking_unconfirmed"
    if (tracking.get("reason") != "rgb_discriminative"
            or tracking.get("positive_patches", 0) < 2
            or tracking.get("negative_patches", 0) < 2
            or float(tracking.get("visual_gain", 0.0)) <= 0):
        return False, "anchor_rgb_not_discriminative_this_frame"
    return True, "provisional_anchor_rgb_not_target_identity"


class UnifiedPGVBDirectTargetPolicy(UnifiedPGVBNamedLandmarkPolicy):
    def __init__(self, stage1, original_v2, trained, directory, *, method='RGB'):
        super().__init__(stage1, original_v2, directory, mode='tracked-stable')
        payload=torch.load(trained,map_location='cpu',weights_only=True)
        if payload.get('format')!='unified-pgvb-direct-target-v1' or payload.get('direct_target_ready') is not True:
            raise ValueError('A fully trained target checkpoint is required')
        if payload['training_config']['source_sha256']!=file_sha256(original_v2):
            raise ValueError('Target model and original integrated source differ')
        self.unified=load_model(payload).eval().requires_grad_(False)
        if method not in ('Geometry','Relation','RGB'):raise ValueError('Unknown direct-target ablation')
        self.method=method

    def set_direct_episode(self, episode, transform, catalog, public_task):
        if public_task['instruction']!=episode.instruction or public_task['map_name']!=episode.map_name:
            raise ValueError('Language-conditioned input mismatch')
        if public_task.get('target_GT_used') is not False:
            raise ValueError('Only public GT-free task specification is allowed')
        # Task whitelist, not a private dataset row or target destination.
        if set(public_task)-{'index','episode','instruction_index','map_name','instruction',
                            'anchors','structured','target_GT_used','world_bounds','start_pose'}:
            raise ValueError('Unexpected metadata in online task')
        self.direct_spec=public_task
        # Cardinal world axes were not calibrated to geographic directions;
        # the 80-query north/south probe failed. Keep these queries in the
        # evaluation denominator but refuse unvalidated semantic displacement.
        self._direction_unverified = any(
            a["predicate"] in ("north", "south", "east", "west")
            for a in public_task["anchors"]
        )
        self.record=public_static_record(episode,transform,catalog,max_landmarks=self.args.max_landmarks)
        self.inputs=None;self.visited_goal_cells=set();self.acquisition.reset();self.stable_goal.reset();self.unified.reset_episode()
        # Several references may be valid for the *relation* head, but the
        # current visual head has no anchor-specific cross-attention over them.
        # Never silently pretend the first polygon is the correct RGB identity.
        anchors = public_task["anchors"]
        if not anchors:
            raise ValueError("A direct-target query needs at least one public anchor")
        public_ids = anchors[0]["landmark"].get("public_ids", ())
        unique_anchor = len(anchors) == 1 and len(public_ids) == 1
        self._visual_anchor_reason = (None if unique_anchor
                                      else "ambiguous_multi_anchor_or_polygon")
        self._landmark = None
        if unique_anchor:
            first = anchors[0]
            self._landmark = dict(name=first["landmark"]["name"],
                                  typ=first["landmark"]["type"],
                                  contours=first["contours"])
        self.instruction_spec=None

    @torch.no_grad()
    def predict(self, state, instruction):
        if instruction != self.record["instruction"]:
            raise ValueError("Instruction changed")
        self._initialize_geometry_inputs()
        start = time.perf_counter()
        observed = meta = tracking = None
        reason = "ablation_no_RGB"
        if self.method == "RGB" and not self._direction_unverified:
            if self._landmark is None:
                reason = self._visual_anchor_reason
            else:
                # The anchor must have a supported current-frame ROI; projected
                # offscreen polygons are not RGB evidence of a named object.
                observed, reason, meta = encode_landmark_observation(
                    state.current_observation, self._landmark, self.record,
                    self.native, self.device, return_metadata=True,
                    allow_offscreen_anchor=False,
                )
        geometry = relation_geometry(
            self.direct_spec, self.inputs["grid_xy"][0].cpu().numpy(),
        )[None].to(self.device)
        ids = torch.tensor(
            [[a["predicate_id"] for a in self.direct_spec["anchors"]]],
            device=self.device,
        )
        category = self.direct_spec["structured"]["target"]["category"]
        semantic = " ".join(
            [category] + [a["value"] for a in
                          self.direct_spec["structured"]["target"]["attributes"]]
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Confirm RGB *before* allowing the target residual to modify the
            # belief. This reuses the already encoded current image patches;
            # only the inexpensive PGVB scoring head runs for counterfactuals.
            if self.method == "RGB":
                if observed is not None:
                    kwargs = {k: v for k, v in observed.items() if k != "local_uv"}
                    scores = self.unified.pgvb(**kwargs)
                    shifted = dict(
                        kwargs,
                        image_features=kwargs["image_features"].roll(
                            shifts=scores.shape[1] // 2, dims=1,
                        ),
                    )
                    counter = self.unified.pgvb(**shifted)
                    tracking = self.acquisition.observe(
                        scores.float(), counter.float(),
                        meta["anchor_labels"], kwargs["valid"],
                    )
                else:
                    tracking = self.acquisition.observe()
            admitted, admission_reason = admissible_target_rgb(
                tracking, meta, observed,
            )
            result = self.unified.forward_direct_target(
                static_inputs=self.inputs, tokens=self.tokens,
                token_mask=self.mask, node_emb=self.names,
                relation_geometry=geometry, relation_ids=ids,
                relation_valid=torch.ones_like(ids, dtype=torch.bool),
                observed=observed if admitted else None,
                target_text=self.native.texts([semantic]) if admitted else None,
            )
        key = {"Geometry": "geometry_field",
               "Relation": "relation_field", "RGB": "field"}[self.method]
        field = (result["geometry_field"] if self._direction_unverified
                 else result[key])
        active = bool(not self._direction_unverified and admitted
                      and result["coverage"].any())
        field, goal_audit = self.stable_goal.select(
            field, self.inputs["grid_xy"],
            tracking=tracking, visual_active=active,
        )
        prediction = A3TrajectoryPolicy.predict_field(self, state, field)
        torch.cuda.synchronize()
        prediction.diagnostics.update(
            method=self.method, RGB_reason=reason,
            uncalibrated_direction_fallback=self._direction_unverified,
            RGB_admitted=bool(active), RGB_admission_reason=(
                "uncalibrated_direction" if self._direction_unverified
                else admission_reason),
            target_identity_independently_verified=False,
            visual_active=active, tracking=tracking,
            stable_goal=goal_audit,
            decision_seconds=time.perf_counter()-start,
            geometry_top1=int(result["geometry_field"].belief_logits.argmax()),
            relation_top1=int(result["relation_field"].belief_logits.argmax()),
            full_top1=int(result["field"].belief_logits.argmax()),
            belief_top1=int(field.belief_logits.argmax()),
        )
        if getattr(self, "direct_audit_callback", None):
            self.direct_audit_callback(state, result, field)
        return prediction
