import copy
import numpy as np
import pytest
from experiments.a2_waldo_landmark_belief.relation_v3_protocol import safe_dev_points,selection_key,verify_history,choose_stage


def population(*,teacher=False,break_teacher=False):
    n=40
    before=np.full(n,10.);after=before.copy()
    if not teacher:before[:6]=30.;after[:6]=10.
    if break_teacher:after[0]=40.
    return dict(scores=np.tile([.2,.7,.1],(n,1)),available=np.ones(n,dtype=bool),before_ne=before,after_ne=after,
        identities=[dict(map_name='teacher' if teacher else 'real',object_id=i) for i in range(n)])


def test_real_dev_gate_must_also_preserve_teacher_using_same_thresholds():
    passed=safe_dev_points(population(),population(teacher=True))
    assert passed['selected_point'] is not None and passed['selected_point']['summary']['gain_groups']==6
    assert safe_dev_points(population(),population(teacher=True,break_teacher=True))['selected_point'] is None


def test_refinement_checkpoint_selection_prioritizes_eligible_real_gate():
    safe=safe_dev_points(population(),population(teacher=True))['selected_point']
    def record(epoch,point,r1):
        return dict(epoch=epoch,dev=dict(selected_point=point,real=dict(proposal_r1=r1,proposal_ne=30.),teacher=dict(proposal_r1=.8,proposal_ne=20.)))
    assert selection_key(record(21,safe,.1))>selection_key(record(22,None,.99))
    assert selection_key(record(1,None,.8))>selection_key(record(2,None,.7))


def test_full_schedule_rejects_short_epochs_or_missing_rows():
    history=[]
    for epoch in range(1,26):
        history.append(dict(epoch=epoch,stage='ranking' if epoch<=20 else 'refinement',stage_epoch=epoch if epoch<=20 else epoch-20,
            train=dict(views=33440,optimizer_updates=4180),dev=dict(views=5647,optimizer_updates=0)))
    verify_history(history)
    with pytest.raises(ValueError):verify_history(history[:-1])
    bad=copy.deepcopy(history);bad[4]['train']['views']-=1
    with pytest.raises(ValueError):verify_history(bad)


def test_corrupt_teacher_scores_cannot_look_like_safe_fallback():
    teacher=population(teacher=True);teacher['scores'][0,1]=np.nan
    with pytest.raises(ValueError):safe_dev_points(population(),teacher)
    teacher=population(teacher=True);teacher['scores']*=2
    with pytest.raises(ValueError):safe_dev_points(population(),teacher)


def test_stage_choice_never_promotes_ranking_checkpoint_when_refinement_has_no_safe_gate():
    def r(epoch,r1):
        return dict(epoch=epoch,dev=dict(selected_point=None,real=dict(proposal_r1=r1,proposal_ne=30.),teacher=dict(proposal_r1=.8,proposal_ne=20.)))
    history=[r(20,.99),r(21,.1)]
    assert choose_stage(history,refine=False)['epoch']==20
    chosen=choose_stage(history,refine=True)
    assert chosen['epoch']==21 and chosen['dev']['selected_point'] is None
