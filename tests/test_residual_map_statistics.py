import numpy as np
import pytest
from scripts.analyze_visual_residual_unseen import paired_bootstrap,pair_ids,public_input_audit


def test_map_cluster_retains_map_sizes_and_reports_two_estimands():
    diff=np.array([1.]+[-1.]*9)
    maps=['small']+['large']*9
    result=paired_bootstrap(diff,maps,replicates=10000,seed=42)
    effect=result['statistics'][0]
    assert result['map_count']==2
    assert effect['episode_weighted_delta_pp']==pytest.approx(-80.)
    assert effect['map_equal_delta_pp']==pytest.approx(0.)
    assert effect['map_cluster_episode_weighted_95ci_pp']==[-100.,100.]
    assert effect['map_cluster_equal_weighted_95ci_pp']==[-100.,100.]
    assert result==paired_bootstrap(diff,maps,replicates=10000,seed=42)


def test_zero_paired_effect_has_degenerate_zero_intervals():
    result=paired_bootstrap(np.zeros((6,2)),['a','a','b','b','c','c'],replicates=10000)
    for e in result['statistics']:
        assert e['episode_weighted_delta_pp']==e['map_equal_delta_pp']==0
        for k in ('episode_paired_95ci_pp','map_cluster_episode_weighted_95ci_pp','map_cluster_equal_weighted_95ci_pp'):
            assert e[k]==[0.,0.]


def row(i,map_name='map'):return dict(episode_id=[map_name,i,0],map_name=map_name,index=i)


def test_pairing_by_episode_id_not_row_position_and_rejects_missing_ids():
    pairs=pair_ids([row(0),row(1)],[row(1),row(0)])
    assert all(a['index']==b['index'] for a,b in pairs)
    with pytest.raises(ValueError):pair_ids([row(0),row(1)],[row(0)])
    with pytest.raises(ValueError):pair_ids([row(0),row(0)],[row(0),row(1)])


def test_private_label_keys_absent_from_network_inference_interfaces():
    audit=public_input_audit()
    assert set(audit['public_episode_fields'])=={'map_name','instruction','start_pose','references','reference_names'}
