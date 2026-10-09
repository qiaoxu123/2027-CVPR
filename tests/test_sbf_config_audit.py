import pytest
from scripts.audit_sbf_configs import configuration_kind


def test_config_audit_distinguishes_model_inheritance_and_eval_suites():
    assert configuration_kind({'_base_':'../main.yaml'}) == 'model'
    assert configuration_kind({'model':{}, 'data':{}}) == 'model'
    assert configuration_kind(dict(seed=0,splits=['val_seen'],episodes_per_split=128,
        variants=[dict(name='base',overrides=[])])) == 'suite'


def test_config_audit_does_not_silently_ignore_unknown_or_test_manifests():
    with pytest.raises(ValueError,match='unknown'):
        configuration_kind(dict(seed=0,variants=[]))
    with pytest.raises(ValueError,match='approved'):
        configuration_kind(dict(seed=0,splits=['test_unseen'],episodes_per_split=128,
            variants=[dict(name='base',overrides=[])]))
