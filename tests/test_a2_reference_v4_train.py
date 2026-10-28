import json
import hashlib
import pytest

from experiments.a2_waldo_landmark_belief.reference_v4_train import validate_complete_feature_cache


def test_training_refuses_missing_or_partial_new_cache(tmp_path):
    with pytest.raises(ValueError,match='Complete reference feature cache'):
        validate_complete_feature_cache(tmp_path)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status='building',entries=[])))
    with pytest.raises(ValueError,match='Complete reference feature cache'):
        validate_complete_feature_cache(tmp_path)


def test_training_refuses_final_marker_without_audited_full_population(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status='all_train_dev_reference_overlays_complete',
        counts={'train':{'observations':1},'dev':{'observations':1}},entries=[],no_gpu_used=True,
        parent_cache_unchanged=True,calibration_tensors_opened=False,benchmark_used=False)))
    with pytest.raises(ValueError,match='population'):
        validate_complete_feature_cache(tmp_path)


def test_training_rejects_resumed_cache_without_trusted_per_file_resume_journal(tmp_path):
    manifest = dict(status='all_train_dev_reference_overlays_complete',
        counts={'train':{'observations':33440},'dev':{'observations':5647}},entries=[],no_gpu_used=True,
        parent_cache_unchanged=True,calibration_tensors_opened=False,benchmark_used=False)
    path = tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    (tmp_path/'status.json').write_text(json.dumps(dict(state='complete',processed=39087,total=39087,
        resumed=1,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    with pytest.raises(ValueError,match='fresh full feature build'):
        validate_complete_feature_cache(tmp_path)
