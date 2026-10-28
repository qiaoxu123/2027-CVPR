import copy
import math
import pytest
from experiments.a2_waldo_landmark_belief.relation_v3_reporting import compare_pairs


def row(index,*,target='x',instruction=None,sr=0,stop=0,false=0,recall=0,updated=False,map_name='map1'):
    return dict(index=index,object_id=target,map_name=map_name,instruction=instruction or f'query{index}',
        metrics=dict(SR=sr,OSR=sr,SPL=sr*.8,NE=30-20*sr,nDTW=.1+.4*sr,sDTW=.5*sr,policy_stop=stop,false_stop=false),
        recall={'dynamic_final_R@1/20m':recall,'dynamic_final_NE_m':30-20*recall},
        step_diagnostics=[dict(a2_updated=updated,a2_candidates=3)],inference_seconds=[.01],
        coverage=[dict(target_visible=True,candidate_point_covered=updated,candidate_within20m=updated)])


def test_conditional_false_stop_has_its_own_denominator_and_pairs_align_by_source_index():
    a0=[row(0,stop=1,false=1),row(1),row(2)]
    a2=[row(0,stop=1,false=1),row(1,stop=1,sr=1,updated=True),row(2,stop=1,sr=1,updated=True)]
    result=compare_pairs(a0,list(reversed(a2)),draws=200)
    rate=result['navigation']['False_Stop_Rate']
    assert rate['A0']==1 and rate['A2']==pytest.approx(1/3)
    assert rate['difference']==pytest.approx(-2/3)
    assert result['navigation']['false_stop_episode_fraction']['difference']==0
    assert result['SR_pairs']['corrected']==2 and result['SR_pairs']['broken']==0
    assert result['mechanism']['fallback_episode_fraction']==pytest.approx(1/3)
    assert result['mechanism']['updated_subset']['episodes']==2
    assert result['mechanism']['updated_subset']['descriptive_only'] is True
    assert result['candidate_coverage']['point_coverage']==pytest.approx(2/3)


def test_no_stop_is_undefined_not_zero_and_invalid_bootstrap_draws_are_disclosed():
    result=compare_pairs([row(0),row(1)],[row(0),row(1)],draws=100)
    rate=result['navigation']['False_Stop_Rate']
    assert rate['A0'] is None and rate['difference'] is None
    assert rate['target_cluster_bootstrap95'] is None
    assert rate['target_cluster_valid_draws']==0
    assert result['navigation']['SR']['target_cluster_bootstrap95']==[0,0]


def test_recall_uses_first_source_query_once_even_when_multiple_episodes_repeat_it():
    a0=[row(0,instruction='same',recall=0),row(1,instruction='same',recall=1),row(2,instruction='other',recall=1)]
    a2=[row(0,instruction='same',recall=1),row(1,instruction='same',recall=0),row(2,instruction='other',recall=1)]
    result=compare_pairs(a0,a2,draws=100)
    assert result['unique_queries']==2
    assert result['recall']['dynamic_final_R@1/20m']['A0']==.5
    assert result['recall']['dynamic_final_R@1/20m']['difference']==.5
    assert result['target_clusters']==1


def test_target_cluster_resampling_keeps_all_episodes_and_uses_episode_weighted_estimand():
    a0=[row(i,target='large' if i<3 else 'small') for i in range(4)]
    a2=[row(i,target='large' if i<3 else 'small',sr=int(i<3)) for i in range(4)]
    result=compare_pairs(a0,a2,draws=200,seed=0)
    value=result['navigation']['SR']
    assert value['difference']==.75
    assert value['target_cluster_bootstrap95']==[0,1]
    assert value['map_cluster_bootstrap95'] is None
    assert compare_pairs(a0,a2,draws=200,seed=0)==result


@pytest.mark.parametrize('mutation',['duplicate','wrong_target','wrong_instruction','missing','nan','metric_schema','nonbinary','false_without_stop'])
def test_invalid_or_unpaired_populations_are_rejected(mutation):
    a0=[row(0),row(1)];a2=copy.deepcopy(a0)
    if mutation=='duplicate':a2[1]['index']=0
    elif mutation=='wrong_target':a2[0]['object_id']='other'
    elif mutation=='wrong_instruction':a2[0]['instruction']='different'
    elif mutation=='missing':a2.pop()
    elif mutation=='nan':a2[0]['metrics']['NE']=math.nan
    elif mutation=='metric_schema':del a2[1]['recall']['dynamic_final_NE_m']
    elif mutation=='nonbinary':a2[0]['metrics']['SR']=.3
    elif mutation=='false_without_stop':a2[0]['metrics']['false_stop']=1
    with pytest.raises(ValueError):compare_pairs(a0,a2,draws=100)


def test_per_map_populations_remain_paired_and_latency_uses_all_actual_decisions():
    a0=[row(0,map_name='one'),row(1,map_name='two')];a2=copy.deepcopy(a0)
    a2[1]['inference_seconds']=[.02,.04];a2[1]['step_diagnostics']*=2;a2[1]['coverage']*=2
    result=compare_pairs(a0,a2,draws=100)
    assert set(result['by_map'])=={'one','two'}
    assert result['by_map']['one']['episodes']==1
    assert result['latency']['A2']['decisions']==3
    assert result['latency']['A2']['mean_ms']==pytest.approx(70/3)
    assert result['navigation']['SR']['map_cluster_bootstrap95']==[0,0]
