"""Episode-local navigation Goal stabilization for a dynamic visual GoalField.

The RGB-modified dense Belief is always preserved. Only candidate order to the
frozen controller is held until consecutive *observed* evidence supports a
different goal. No CityNav target labels, public-polygon oracle goal or Stop
threshold enters this module. The prior 1 mm controller arrival guard remains.
"""
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class StableGoalConfig:
    # Initial engineering safeguards, NOT tuned on the frozen Val-Seen64.
    replan_logit_margin: float = 0.35
    replan_confirmations: int = 2

    def __post_init__(self):
        if self.replan_logit_margin < 0:
            raise ValueError("Replanning margin must be nonnegative")
        if self.replan_confirmations < 2:
            raise ValueError("A dynamic Goal requires repeated confirmation")


class StableNavigationGoal:
    """Keep controller target stable while UnifiedPGVB updates every frame."""

    def __init__(self, config=None):
        self.config = config or StableGoalConfig()
        self.reset()

    def reset(self):
        self.goal_id = None
        self.pending_id = None
        self.pending_count = 0
        self.replans = 0
        self.decisions = 0

    def select(self, field, grid_xy, *, tracking=None, visual_active=False):
        if (
            field.belief_logits.ndim != 2
            or field.belief_logits.shape[0] != 1
            or field.candidate_ids.ndim != 2
            or field.candidate_ids.shape[0] != 1
            or grid_xy.shape != (*field.belief_logits.shape, 2)
            or field.dense_context is None
            or field.dense_context.shape[:2] != field.belief_logits.shape
        ):
            raise ValueError("Stable Goal requires a single complete dense GoalField")
        if not torch.isfinite(field.belief_logits).all():
            raise ValueError("Nonfinite GoalField logits")
        k = field.candidate_ids.shape[1]
        if k < 1:
            raise ValueError("At least one candidate is required")
        best_id = int(field.candidate_ids[0, 0])
        if not 0 <= best_id < field.belief_logits.shape[1]:
            raise ValueError("Out-of-bounds Goal candidate")
        self.decisions += 1
        event = "kept"
        replan_block_reason = "none"
        visual_ready = bool(
            visual_active and tracking is not None
            and tracking.get("state") == "tracking"
            and float(tracking.get("visual_gain", 0.0)) > 0
        )
        advantage = 0.0
        if self.goal_id is None:
            self.goal_id = best_id
            event = "initialized"
        else:
            candidate_covered = bool(
                field.visual_coverage is not None
                and field.visual_coverage[0, best_id] > 0
            )
            strong_visual = visual_ready and candidate_covered
            advantage = float(
                (field.belief_logits[0, best_id]
                 - field.belief_logits[0, self.goal_id]).detach()
            )
            if best_id == self.goal_id:
                replan_block_reason = "same_goal"
            elif not visual_ready:
                replan_block_reason = "no_current_confirmed_visual_evidence"
            elif not candidate_covered:
                replan_block_reason = "new_goal_outside_observed_support"
            elif advantage < self.config.replan_logit_margin:
                replan_block_reason = "insufficient_logit_advantage"
            if (best_id != self.goal_id and strong_visual
                    and advantage >= self.config.replan_logit_margin):
                if self.pending_id == best_id:
                    self.pending_count += 1
                else:
                    self.pending_id = best_id
                    self.pending_count = 1
                event = "pending"
                replan_block_reason = "awaiting_repeated_confirmation"
                if self.pending_count >= self.config.replan_confirmations:
                    self.goal_id = best_id
                    self.replans += 1
                    self.pending_id = None
                    self.pending_count = 0
                    event = "replanned"
                    replan_block_reason = "none"
            else:
                self.pending_id = None
                self.pending_count = 0

        chosen = int(self.goal_id)
        current = field.candidate_ids[0]
        if int(current[0]) == chosen:
            controlled = field
        else:
            # Insert held world-cell candidate at index zero. Rebuild all
            # candidate_xy/context/logits from that same ID; never mismatch.
            remaining = current[current != chosen][:k - 1]
            if remaining.numel() != k - 1:
                raise ValueError("Candidate set contains duplicate indices")
            ids = torch.cat((current.new_tensor([chosen]), remaining))[None]
            context = field.dense_context.gather(
                1, ids[..., None].expand(-1, -1, field.dense_context.shape[-1]),
            )
            xy = grid_xy.gather(1, ids[..., None].expand(-1, -1, 2))
            controlled = replace(
                field, candidate_ids=ids, candidate_xy=xy,
                candidate_context=context,
            )
        return controlled, dict(
            event=event,
            visual_top1_id=best_id,
            held_goal_id=chosen,
            held_goal_present_in_visual_topk=bool((current == chosen).any()),
            pending_goal_id=self.pending_id,
            pending_count=self.pending_count,
            replan_count=self.replans,
            decisions=self.decisions,
            preserves_dense_visual_belief=True,
            stop_guard_unchanged=True,
            replan_block_reason=replan_block_reason,
            replan_logit_advantage=advantage,
            confirmed_visual_evidence=visual_ready,
        )
