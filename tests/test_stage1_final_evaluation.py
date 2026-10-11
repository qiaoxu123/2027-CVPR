import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.evaluate_closedloop import validate_stage1_frozen_evaluation


def test_test_unseen_requires_completed_frozen_candidate_and_unchanged_controls(tmp_path):
    checkpoint = tmp_path/'best.pt'
    checkpoint.write_bytes(b'fixed checkpoint')
    manifest = tmp_path/'frozen.json'
    frozen = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  all_stages_complete=True, completed_epochs=dict(A=3, B=3, C=5),
                  visual_control='real', history_control='checkpoint',
                  stop_threshold=.3, active_observation=False)
    manifest.write_text(json.dumps(frozen))
    args = SimpleNamespace(split='test_unseen', frozen_final_candidate=str(manifest),
        checkpoint=str(checkpoint), stage1_visual_control='real',
        stage1_history_control='checkpoint', stop_threshold=.3, max_episodes=None)
    validate_stage1_frozen_evaluation(args)
    args.max_episodes = 128
    with pytest.raises(ValueError, match='without episode subsampling'):
        validate_stage1_frozen_evaluation(args)
    args.max_episodes = None
    args.stage1_visual_control = 'zero'
    with pytest.raises(ValueError, match='inference parameters'):
        validate_stage1_frozen_evaluation(args)
    args.stage1_visual_control = 'real'
    frozen['completed_epochs']['C'] = 4
    manifest.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='Full Stage1'):
        validate_stage1_frozen_evaluation(args)
    frozen['completed_epochs']['C'] = 5
    manifest.write_text(json.dumps(frozen))
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='checkpoint differs'):
        validate_stage1_frozen_evaluation(args)
