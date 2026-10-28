import pytest
from experiments.a2_waldo_landmark_belief.revision_paired_stats import paired_statistics


def row(index,sr,ne,group='a'):
    return dict(index=index,map_name='map',object_id=group,metrics=dict(SR=sr,NE=ne))


def test_paired_stats_preserves_episode_weights_and_reports_discordant_pairs():
    a=[row(0,0.,30.),row(1,0.,30.),row(2,1.,10.,'b')]
    b=[row(0,1.,10.),row(1,1.,10.),row(2,0.,30.,'b')]
    r=paired_statistics(a,b,draws=1000)
    assert r['metrics']['SR']['difference']==pytest.approx(1/3)
    assert r['metrics']['NE']['difference']==pytest.approx(-20/3)
    assert r['SR_pairs']['corrected']==2 and r['SR_pairs']['broken']==1
    assert r['SR_pairs']['mcnemar_exact_p']==1.
    assert r['target_clusters']==2
    assert r['metrics']['SR']['cluster_bootstrap95']==[-1.,1.]


def test_identical_paired_outcomes_have_exact_zero_intervals():
    a=[row(i,float(i%2),20.) for i in range(6)]
    r=paired_statistics(a,a,draws=100)
    assert r['metrics']['SR']['cluster_bootstrap95']==[0.,0.]
    assert r['metrics']['NE']['difference']==0.
    assert r['SR_pairs']['mcnemar_exact_p']==1.


def test_statistics_reject_missing_or_misaligned_episode_identity():
    a=[row(0,0.,30.)]
    with pytest.raises(ValueError):paired_statistics(a,[])
    with pytest.raises(ValueError):paired_statistics(a,[row(1,1.,10.)])
    with pytest.raises(ValueError):paired_statistics(a,[row(0,1.,10.,'different')])
