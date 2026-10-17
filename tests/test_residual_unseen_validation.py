import hashlib,json
from types import SimpleNamespace
import pytest
from scripts.evaluate_closedloop import validate_stage1_frozen_evaluation


def setup(tmp_path):
    base=tmp_path/'base.pt';base.write_bytes(b'base')
    residual=tmp_path/'residual.pt';residual.write_bytes(b'residual')
    frozen=tmp_path/'frozen.json'
    manifest=dict(validation_kind='calibrated-residual-val-unseen-v1',
                  checkpoint_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                  calibrated_residual_sha256=hashlib.sha256(residual.read_bytes()).hexdigest(),
                  all_stages_complete=True,completed_epochs=dict(A=3,B=3,C=5),grid_size=28,
                  allowed_visual_controls=['real','no-belief','zero','shuffle'],
                  visual_control='real',history_control='checkpoint',stop_threshold=.3,active_observation=False)
    frozen.write_text(json.dumps(manifest))
    args=SimpleNamespace(split='val_unseen',checkpoint=str(base),stage1_residual_checkpoint=str(residual),
                         frozen_final_candidate=str(frozen),stage1_grid_size=None,stage1_resolution_eval=False,
                         max_episodes=None,stage1_visual_control='real',stage1_history_control='checkpoint',stop_threshold=.3)
    return args,manifest,frozen


@pytest.mark.parametrize('control',['real','no-belief','zero','shuffle'])
def test_frozen_unseen_controls_allowed(tmp_path,control):
    args,_,_=setup(tmp_path);args.stage1_visual_control=control
    validate_stage1_frozen_evaluation(args)


@pytest.mark.parametrize('change',[
    dict(max_episodes=128),dict(stage1_grid_size=384),dict(stop_threshold=.5),
    dict(stage1_history_control='current'),dict(stage1_resolution_eval=True)])
def test_unseen_validation_rejects_configuration_changes(tmp_path,change):
    args,_,_=setup(tmp_path)
    for k,v in change.items():setattr(args,k,v)
    with pytest.raises(ValueError):validate_stage1_frozen_evaluation(args)


def test_replaced_weights_and_legacy_manifest_rejected(tmp_path):
    args,manifest,frozen=setup(tmp_path)
    manifest.pop('validation_kind');frozen.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):validate_stage1_frozen_evaluation(args)
    args,manifest,frozen=setup(tmp_path)
    from pathlib import Path
    Path(args.stage1_residual_checkpoint).write_bytes(b'changed')
    with pytest.raises(ValueError):validate_stage1_frozen_evaluation(args)


def test_original_formal_control_protection_unchanged(tmp_path):
    args,manifest,frozen=setup(tmp_path);args.stage1_residual_checkpoint=None
    args.stage1_visual_control='shuffle'
    with pytest.raises(ValueError):validate_stage1_frozen_evaluation(args)
