import json
from types import SimpleNamespace
import pytest
from scripts.run_joint_rgb_geometry_core import candidate_dev_recall
from scripts.evaluate_closedloop import validate_stage1_frozen_evaluation


def test_selection_uses_selected_epoch_instead_of_last_epoch():
    state = dict(best_key=[.5, .4, .7], history=[dict(epoch=1, dev={'belief_R@1/20m': .7}), dict(epoch=2, dev={'belief_R@1/20m': .2})])
    assert candidate_dev_recall(state) == .7


def test_joint_heldout_requires_same_complete_protocol(tmp_path):
    ck = tmp_path/'best.pt'; ck.write_bytes(b'checkpoint')
    from scripts.evaluate_closedloop import sha
    frozen = dict(checkpoint_sha256=sha(ck), all_stages_complete=True, completed_epochs=dict(A=3,B=3,C=5), visual_control='real', history_control='checkpoint', stop_threshold=.3, active_observation=False, model='graphnav-joint-rgb', joint_training_complete=True)
    fp=tmp_path/'freeze.json';fp.write_text(json.dumps(frozen))
    a=SimpleNamespace(split='val_unseen', frozen_final_candidate=str(fp), checkpoint=str(ck), stage1_visual_control='real', stage1_history_control='checkpoint', stop_threshold=.3, max_episodes=None, model='graphnav-joint-rgb')
    validate_stage1_frozen_evaluation(a)
    frozen['completed_epochs']['C']=4;fp.write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match='Full Stage1'):
        validate_stage1_frozen_evaluation(a)


def test_paired_statistics_preserves_pairing_and_stop_denominators():
    from scripts.summarize_core_pair import paired_statistics
    def row(i, success, stop):
        return dict(index=i,episode_id=['m',i],map_name='m',public_query_sha256=str(i),metrics=dict(SR=success,OSR=success,SPL=success,NE=10.,nDTW=.5,sDTW=.5*success,policy_stop=stop,false_stop=stop*(1-success)))
    left=[row(0,0,1),row(1,1,1)]
    right=[row(0,1,1),row(1,1,0)]
    r=paired_statistics(left,right,repetitions=100)
    assert r['SR']['A1_minus_A0']==.5
    assert r['False_Stop_Rate']['A1_minus_A0']==-.5
    assert r['SR_McNemar_exact']['A1_only_success']==1
    with pytest.raises(ValueError,match='identity mismatch'):
        paired_statistics(left,right[::-1],repetitions=10)


def test_paired_recall_uses_unique_queries():
    from scripts.summarize_core_pair import paired_recall
    a=[dict(public_query_sha256='q',goal_retrieval={'dynamic_final_R@1/20m':0.}),dict(public_query_sha256='q',goal_retrieval={'dynamic_final_R@1/20m':1.})]
    b=[dict(public_query_sha256='q',goal_retrieval={'dynamic_final_R@1/20m':1.}),dict(public_query_sha256='q',goal_retrieval={'dynamic_final_R@1/20m':0.})]
    r=paired_recall(a,b,repetitions=20)
    assert r['unique_queries']==1
    assert r['dynamic_final_R@1/20m']['A1_minus_A0']==1.


def test_frozen_evaluation_rejects_changed_controller_yaml(tmp_path):
    from scripts.evaluate_closedloop import sha
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'fixed')
    config=tmp_path/'config.yaml';config.write_text('navigation: original')
    frozen=dict(checkpoint_sha256=sha(checkpoint),all_stages_complete=True,completed_epochs=dict(A=3,B=3,C=5),visual_control='real',history_control='checkpoint',stop_threshold=.3,active_observation=False,sbf_config_sha256=sha(config))
    path=tmp_path/'freeze.json';path.write_text(json.dumps(frozen))
    a=SimpleNamespace(split='val_unseen',frozen_final_candidate=str(path),checkpoint=str(checkpoint),stage1_visual_control='real',stage1_history_control='checkpoint',stop_threshold=.3,max_episodes=None,config=str(config))
    validate_stage1_frozen_evaluation(a)
    config.write_text('navigation: changed')
    with pytest.raises(ValueError,match='controller config'):
        validate_stage1_frozen_evaluation(a)


def test_report_formats_absent_policy_stop_rate():
    from scripts.summarize_core_pair import format_metric
    assert format_metric(None)=='N/A'
    assert format_metric(.5)=='0.500000'


def test_a1_only_schedule_does_not_include_A0():
    from scripts.run_joint_rgb_geometry_core import scheduled_arms
    assert [a[0]for a in scheduled_arms(True)]==['A1_joint_rgb_geometry']
    assert len(scheduled_arms(False))==2


def test_historical_recall_rejects_live_label_edits(tmp_path):
    from scripts.summarize_core_pair import validate_report_inputs
    from graphnav.checkpointing import file_sha256
    annotation=tmp_path/'val_seen.json';annotation.write_text('original labels')
    expected={'old-host/val_seen.json':file_sha256(annotation)}
    validate_report_inputs([annotation],expected)
    annotation.write_text('changed labels, identical episode IDs')
    with pytest.raises(ValueError,match='live source data'):
        validate_report_inputs([annotation],expected)
