import copy
import pytest
from experiments.a2_waldo_landmark_belief.relation_v3_preflight import check_records


def rows():
    return [dict(index=i,step=0,should_stop=i==0,xyz=[0.,0.,0.],selected_goal_xy=[0.,0.],belief_top16_world_xy=[[0.,0.]],
        belief_logits_sha256='x',belief_logits_shape=[1,784],belief_logits_dtype='torch.float32',a2_updated=False) for i in range(2)]


def test_functional_preflight_counts_effective_actions_and_requires_exact_off():
    a=rows();off=copy.deepcopy(a);on=copy.deepcopy(a)
    for r in on:r.update(a2_updated=True,selected_goal_xy=[10.,0.],xyz=[10.,0.,0.])
    result=check_records(a,off,on)
    assert result['changed_actions']==1 and result['changed_fields']==2
    off[0]['selected_goal_xy']=[1.,0.]
    with pytest.raises(ValueError):check_records(a,off,on)


def test_stop_coordinate_changes_alone_do_not_prove_controller_coupling():
    a=rows()[:1];off=copy.deepcopy(a);on=copy.deepcopy(a)
    on[0].update(a2_updated=True,selected_goal_xy=[10.,0.],xyz=[10.,0.,0.])
    with pytest.raises(ValueError,match='coupling'):check_records(a,off,on)


def test_OFF_rejects_changed_belief_values_with_identical_rankings_and_actions():
    a=rows();off=copy.deepcopy(a);on=copy.deepcopy(a)
    off[0]['belief_logits_sha256']='different'
    with pytest.raises(ValueError,match='OFF'):check_records(a,off,on)
