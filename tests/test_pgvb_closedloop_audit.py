"""Full per-step evidence capture is opt-in in the unified rollout runner."""
from types import SimpleNamespace

import numpy as np

import torch

from graphnav.full_localization import GoalField
from graphnav.pgvb_closedloop_audit import DecisionRecorder, pack_field


def test_pack_field_is_detached_from_live_goalfield():
    logits = torch.tensor([[0., 1., 2., 3.]])
    dense = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    ids = torch.tensor([[3, 2]])
    xy = torch.tensor([[[.1, .2], [.3, .4]]])
    field = GoalField(
        belief_logits=logits, candidate_ids=ids, candidate_xy=xy,
        candidate_context=dense.gather(
            1, ids[..., None].expand(-1, -1, dense.shape[-1]),
        ),
        dense_context=dense,
    )
    before = field.belief_logits.clone()
    packed = pack_field(field)
    packed["logits"][0] = 99.
    assert torch.equal(field.belief_logits, before)
    assert packed["candidate_context"].shape == (2, 2)


class FakePolicy:
    def __init__(self):
        self.audit_callback = None
        self.record = {"world_bounds": [0., 0., 100., 100.]}

    def predict(self, state, instruction):
        if self.audit_callback is not None:
            self.audit_callback(None, None, None)
        return SimpleNamespace(diagnostics={"instruction": instruction})


def test_lightweight_recorder_skips_all_per_step_artifacts(tmp_path):
    policy = FakePolicy()
    output = tmp_path / "decisions"
    recorder = DecisionRecorder(policy, output, save_artifacts=False)
    assert policy.audit_callback is None
    prediction = recorder.predict(SimpleNamespace(), "Navigate to landmark")
    assert prediction.diagnostics == {"instruction": "Navigate to landmark"}
    assert not output.exists()


def test_full_capture_requires_opt_in_and_can_be_cleared(tmp_path):
    policy = FakePolicy()
    directory = tmp_path / "decisions"
    full = DecisionRecorder(policy, directory, save_artifacts=True)
    assert policy.audit_callback is not None
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    obs = {"bgr": rgb, "footprint_world": np.zeros((4, 2))}
    state = SimpleNamespace(
        current_observation=obs, pose=[1., 2., 30.], step=1,
    )
    result = full.predict(state, "Navigate")
    assert (directory / "step_01.npz").exists()
    assert (directory / "step_01.png").exists()
    assert result.diagnostics["audit_artifact"] == str(directory / "step_01.npz")
    lite = DecisionRecorder(policy, directory, save_artifacts=False)
    assert policy.audit_callback is None
    lite.predict(SimpleNamespace(), "Navigate")
    assert sorted(p.name for p in directory.iterdir()) == [
        "step_01.npz", "step_01.png",
    ]
