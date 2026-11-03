import pytest
from scripts.report_direct_relation_navigation import aggregate


def row(i, success):
    return dict(index=i,map_name='m',metrics=dict(SR=success,OSR=1,SPL=success*.8,NE=10 if success else 30,nDTW=.2))


def test_navigation_success_counts_and_denominator():
    result=aggregate([row(1,1),row(2,0),row(3,1)])
    assert result['episodes']==3 and result['successes']==2
    assert result['SR']==2/3
    assert result['NE']==50/3
    assert result['sr_wilson95'][0]<result['SR']<result['sr_wilson95'][1]


def test_duplicate_or_nonbinary_episode_results_are_rejected():
    with pytest.raises(ValueError):
        aggregate([row(1,1),row(1,0)])
    with pytest.raises(ValueError):
        aggregate([row(1,.5)])
