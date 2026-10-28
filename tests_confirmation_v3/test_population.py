"""Separate post-training-stage tests; frozen training test inventory is untouched."""
import copy,hashlib
import pytest
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold
from experiments.a2_waldo_landmark_belief.relation_v3_confirmation_protocol import build_population


def fixture():
    fit=next(i for i in range(100) if calibration_fold('m',i)=='fit')
    verify=[i for i in range(100) if calibration_fold('m',i)=='verify'][:2]
    targets=[fit,*verify];entries=[];fresh=[]
    for k,target in enumerate(targets):
        indices=list(range(2*k,2*k+2))
        for i in indices:entries.append(dict(index=i,map_name='m',object_id=target,description_id=i,split='dev'))
        if k<2:
            selected=min(indices,key=lambda i:hashlib.sha256(f'relation-v3-query:0:{i}'.encode()).digest())
            fresh.append(dict(map_name='m',object_id=target,episode_indices=indices,index=selected,fold='fit' if k==0 else 'verify'))
    role=lambda values:[dict(map_name='m',object_id=i) for i in values]
    roles=dict(target_roles=dict(train=role([1000]),dev=role([1001]),calibration_fit=role([fit]),calibration_verify=role(verify)),fresh_calibration=fresh)
    sealed=dict(targets=[dict(entries[4],fold='verify')])
    return roles,dict(entries=entries),sealed


COUNTS=dict(fresh=2,fit=1,verify=2,sealed=1)


def test_population_preserves_predeclared_queries_and_disjoint_folds():
    r,s,sealed=fixture();result=build_population(r,s,sealed,expected=COUNTS)
    assert len(result['fresh_entries'])==2 and len(result['sealed_entries'])==1
    assert [e['index'] for e in result['fresh_entries']]==[e['index'] for e in r['fresh_calibration']]
    assert len({(e['map_name'],e['object_id']) for e in result['fresh_entries']+result['sealed_entries']})==3


def test_supervised_or_sealed_targets_cannot_become_fresh_confirmation():
    r,s,sealed=fixture();bad=copy.deepcopy(r);bad['target_roles']['train'].append(bad['target_roles']['calibration_fit'][0])
    with pytest.raises(ValueError):build_population(bad,s,sealed,expected=COUNTS)
    bad=copy.deepcopy(r);bad['fresh_calibration'][1]['object_id']=sealed['targets'][0]['object_id']
    with pytest.raises(ValueError):build_population(bad,s,sealed,expected=COUNTS)


def test_query_reselection_or_fold_change_is_rejected():
    r,s,sealed=fixture();bad=copy.deepcopy(r);fresh=bad['fresh_calibration'][0]
    fresh['index']=next(i for i in fresh['episode_indices'] if i!=fresh['index'])
    with pytest.raises(ValueError):build_population(bad,s,sealed,expected=COUNTS)
    bad=copy.deepcopy(r);bad['fresh_calibration'][0]['fold']='verify'
    with pytest.raises(ValueError):build_population(bad,s,sealed,expected=COUNTS)
