import numpy as np
import pytest
from scripts.polygon_ablation_statistics import paired_ci,paired_rows
from scripts.polygon_ablation_plan import canonical_job


def test_shared_full_run_is_exact_alias_without_merging_other_representations():
    for seed in (0,1,2):
        assert canonical_job('descriptor_only','full_polygon',seed)==canonical_job('end_to_end','full_polygon',seed)
        assert canonical_job('descriptor_only','centroid',seed)!=canonical_job('end_to_end','centroid',seed)


def test_identical_pairs_give_exact_zero_both_bootstrap_levels():
    r=paired_ci(np.zeros((6,2)),['a']*4+['b']*2,['R@1','SR'],draws=1000)
    assert r['independent_maps']==2
    for v in r['metrics'].values():
        assert v['delta']==0 and v['map_equal_delta']==0
        assert v['episode_ci95']==v['map_cluster_episode_weighted_ci95']==v['map_cluster_map_equal_ci95']==[0.,0.]


def test_cluster_estimands_and_leave_one_map_out_are_distinct():
    r=paired_ci([1.,1.,-1.],['large','large','small'],['SR'],draws=1000)
    assert r['metrics']['SR']['delta']==pytest.approx(1/3)
    assert r['metrics']['SR']['map_equal_delta']==0
    assert {x['excluded_map']:x['episode_weighted']['SR'] for x in r['leave_one_map_out']}=={'large':-1.,'small':1.}


def test_pairing_uses_ids_and_rejects_mismatched_queries():
    a=[dict(query_id='x',map_name='m',**{'R@1':1.}),dict(query_id='y',map_name='m',**{'R@1':0.})]
    b=[dict(query_id='y',map_name='m',**{'R@1':1.}),dict(query_id='x',map_name='m',**{'R@1':0.})]
    d,m,k=paired_rows(a,b,'localization',['R@1'])
    assert d[:,0].tolist()==[1.,-1.] and k==['x','y']
    with pytest.raises(AssertionError):paired_rows(a,b[:1],'localization',['R@1'])
