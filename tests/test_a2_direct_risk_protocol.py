import numpy as np
import pytest
from experiments.a2_waldo_landmark_belief.direct_risk_protocol import group_effects,operating_points,select_point,point_key


def test_effects_use_target_equal_means_and_original_correct_group_denominator():
    ids=[dict(map_name='m',object_id=i) for i in (0,0,1)]
    r=group_effects(ids,np.array([40.,10.,40.]),np.array([10.,50.,10.]),np.ones(3,bool))
    assert r['mean_r1_difference']==.5 and r['corrected']==2 and r['broken']==1
    assert r['gain_groups']==1 and r['broken_group_rate']==1.


def test_thresholds_depend_on_available_scores_only_and_empty_is_OFF():
    ids=[dict(map_name='m',object_id=i) for i in range(4)]
    scores=np.array([[.1,.8,.1],[.1,.7,.2],[.1,.6,.3],[0.,0.,1.]])
    points=operating_points(scores,np.array([1,1,1,0],bool),ids,np.full(4,40.),np.full(4,10.))
    assert len(points)==30
    assert points[0]['parameters']['minimum_gain']==pytest.approx(.6)
    assert points[0]['parameters']['maximum_harm']==pytest.approx(.11)
    assert points[0]['summary']['supported_groups']==1
    assert operating_points(scores,np.zeros(4,bool),ids,np.full(4,40.),np.full(4,10.))==[]


def test_select_point_cannot_choose_harmful_or_zero_benefit_configuration():
    def p(index,gain,broken,ne):
        return dict(index=index,gain_quantile=.5,harm_quantile=.1,parameters={},summary=dict(mean_r1_difference=gain,
            broken_groups=broken,mean_ne_difference_m=ne,supported_groups=40,gain_groups=6,broken_group_rate=broken/100))
    safe=p(0,.02,0,-1.);unsafe=p(1,.5,10,-2.);zero=p(2,0.,0,-1.)
    assert select_point([safe,unsafe,zero],dev=True)==safe
    assert select_point([unsafe,zero],dev=True) is None
    assert point_key(safe)>point_key(p(1,.02,1,-1.))
