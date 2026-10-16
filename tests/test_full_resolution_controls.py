import hashlib,json
from types import SimpleNamespace
import pytest
from scripts.evaluate_closedloop import configure_stage1_controls,validate_stage1_frozen_evaluation


def test_test_resolution_requires_a_separately_frozen_grid_and_full_episode_set(tmp_path):
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'fixed')
    frozen=tmp_path/'frozen.json'
    values=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),all_stages_complete=True,
        completed_epochs=dict(A=3,B=3,C=5),visual_control='real',history_control='checkpoint',stop_threshold=.3,active_observation=False)
    frozen.write_text(json.dumps(values))
    args=SimpleNamespace(stage1_resolution_eval=True,stage1_diagnostic=False,split='test_unseen',stage1_grid_size=384,
        stage1_visual_control='real',stage1_history_control='checkpoint',stop_threshold=.7,checkpoint=str(checkpoint),
        frozen_final_candidate=str(frozen),max_episodes=None)
    configure_stage1_controls(args);assert args.stop_threshold==.3
    with pytest.raises(ValueError,match='separately frozen'):validate_stage1_frozen_evaluation(args)
    values['grid_size']=384;frozen.write_text(json.dumps(values));validate_stage1_frozen_evaluation(args)
    args.max_episodes=10
    with pytest.raises(ValueError,match='without episode'):validate_stage1_frozen_evaluation(args)
