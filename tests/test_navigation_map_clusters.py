from scripts.summarize_navigation_map_clusters import cluster_effect


def test_map_weighting_does_not_treat_many_correlated_episodes_as_many_maps():
    result=cluster_effect([1.]+[-1.]*9,['small']+['large']*9,repeats=1000)
    assert abs(result['episode_weighted_difference']+.8)<1e-8
    assert result['map_equal_difference']==0.
    assert result['map_cluster_95ci_map_equal']==[-1.,1.]
    assert result['by_map']['large']['episodes']==9
