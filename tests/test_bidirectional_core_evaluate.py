import pytest
from experiments.bidirectional_core.evaluate import object_rank,pixel_group,swap_plan,object_metadata,swap_outcomes,audit_row,pair_outcomes
from experiments.bidirectional_core.stats import interval,paired_difference
from experiments.bidirectional_core.report import unseen_gate


def objects():return [dict(object_id='a',object_type='Building'),dict(object_id='b',object_type='Building'),dict(object_id='c',object_type='Car')]


def test_unknown_high_score_is_not_semantic_negative():
    rank=object_rank([.8,.7,99],[1,1,1],[1,0,0],[1,1,0],objects())
    assert rank['r1']==1 and rank['candidate_objects']==2


def test_multi_positive_and_duplicate_views_rank_objects():
    obj=objects()+[dict(object_id='a',object_type='Building')]
    rank=object_rank([.9,.7,.6,.1],[1]*4,[1,1,0,1],[1]*4,obj)
    assert rank['candidate_objects']==3 and rank['positive_objects']==2
    assert rank['top_object_id']=='b' and rank['r1']==1


def test_same_class_rank_requires_judged_negative():
    assert object_rank([.8,.7,.99],[1]*3,[1,0,0],[1,0,1],objects(),same_class=True) is None


def test_swap_is_fixed_same_class_and_different_identity():
    plan=swap_plan([1]*3,[1,0,0],[1]*3,objects())
    assert plan['positive_object_id']=='a' and plan['negative_object_id']=='b'
    assert swap_plan([1]*3,[1,0,0],[1,0,1],objects()) is None


def test_resolution_boundaries_and_unknown():
    assert [pixel_group(x) for x in [None,7,8,23,24,47,48]]==['unknown','below8','low_8_23','low_8_23','medium_24_47','medium_24_47','high_48_plus']


def test_metadata_must_align_exactly():
    with pytest.raises(ValueError,match='length'):object_metadata({'roi_objects':objects()},2)


def row(sample,value):return dict(sample_id=sample,family_id='f'+sample,map_name='map',metrics={'r1':value})


def test_paired_population_and_eligibility_cannot_silently_truncate():
    with pytest.raises(ValueError,match='populations'):paired_difference([row('a',1)],[row('b',1)],'r1')
    with pytest.raises(ValueError,match='eligibility'):paired_difference([row('a',1)],[row('a',None)],'r1')


def test_family_clusters_expose_repeated_state_dependence():
    values=[0.]*20+[1.]*20
    iid=interval(values,repetitions=500);cluster=interval(values,groups=['f0']*20+['f1']*20,repetitions=500)
    assert cluster==[0.,1.] and iid[1]-iid[0]<1


def test_unseen_primary_gate_accepts_directional_E1_navigation_gain():
    result=dict(mean=.1,map_cluster_95ci=[.01,.2])
    nav={'paired_differences':{'E0_to_E1':{'by_direction':{'reverse':{'SR':result}}}}}
    gate=unseen_gate({},nav)
    assert gate['eligible'] and not gate['E2_visual_gain_clear']


def test_unseen_gate_requires_clear_primary_not_secondary_navigation():
    result=dict(mean=.1,map_cluster_95ci=[-.01,.2])
    offline={'paired_differences':{'E1_to_E2':{'weak_attribute_text_region_r1':result}}}
    nav={'paired_differences':{'E1_to_E2':{'SR':dict(mean=.9,map_cluster_95ci=[.8,1])}}}
    assert not unseen_gate(offline,nav)['eligible']


def test_swap_fixed_labels_and_source_tracking_answer_different_questions():
    obj=objects()[:2];r=swap_outcomes([.9,.1],[.1,.9],[1,1],[1,0],[1,1],obj,[0,1])
    assert r['original']['r1']==1 and r['fixed_labels']['r1']==0
    assert r['source_tracking']['r1']==1 and r['real_minus_fixed']==1


def test_future_roi_is_rejected_before_model_input():
    from experiments.bidirectional_core.model import CoreModel
    public={key:0 for key in CoreModel.PUBLIC_KEYS};public['roi_grid_ids']=[0]
    meta=dict(cutoff_step=2,roi_objects=[dict(object_id='a',object_type='Car',observed_step=3,observation_cutoff_step=3)])
    with pytest.raises(ValueError,match='future'):audit_row(dict(public=public,metadata=meta))


def test_extra_private_public_field_is_rejected():
    from experiments.bidirectional_core.model import CoreModel
    public={key:0 for key in CoreModel.PUBLIC_KEYS};public['target_world']=[0,0,0]
    with pytest.raises(ValueError,match='private'):audit_row(dict(public=public,metadata={}))


def test_pair_excludes_other_positive_without_making_it_negative():
    obj=[dict(object_id=k,object_type='Building') for k in ['a','b','c']]
    plan=swap_plan([1]*3,[1,0,1],[1]*3,obj,preferred_positive_id='a')
    assert plan['positive_object_id']=='a'
    result=pair_outcomes([.8,.2,.99],[.2,.8,.99],[1]*3,[1,0,1],[1]*3,obj,plan)
    assert result['original']['candidate_objects']==2
    assert result['fixed_labels']['r1']==0 and result['real_minus_fixed']==1
    assert result['source_tracking']['r1']==1


def test_original_target_must_be_visible_judged_positive_for_swap():
    assert swap_plan([1]*3,[1,0,1],[1]*3,objects(),preferred_positive_id='missing') is None
    assert swap_plan([1]*3,[1,0,1],[1]*3,objects(),preferred_positive_id='b') is None


def test_private_episode_association_cannot_silently_duplicate_or_rebind_map():
    from experiments.bidirectional_core.evaluate import episode_lookup,audit_episode
    episode=dict(episode_id='a',family_id='f',map_name='map',direction='forward',source_episode_index=1)
    with pytest.raises(ValueError,match='duplicate'):episode_lookup([episode,episode])
    bad=dict(episode);bad['map_name']='other'
    with pytest.raises(ValueError,match='mismatch'):audit_episode(bad,episode_lookup([episode]))
