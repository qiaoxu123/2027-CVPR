import torch
from scripts.diagnose_aerialvg_candidate_bottlenecks import selection_stages,oracle_flags


def test_nms_can_suppress_position_correct_candidate_and_budget_is_separate():
    boxes=torch.tensor([[.5,.5,.4,.4],[.55,.5,.4,.4],[.1,.1,.05,.05]])
    scores=torch.tensor([.9,.8,.7]);quality=torch.ones(3);geometry=torch.tensor([0.,0.,0.])
    stages=selection_stages(boxes,scores,quality,geometry)
    assert stages['valid900'].tolist()==[0,1,2]
    assert stages['nms'].tolist()==[0,2]
    distances=torch.tensor([25.,5.,80.])
    assert oracle_flags(distances,stages['valid900'],False)['oracle20']
    assert not oracle_flags(distances,stages['nms'],False)['oracle20']
    assert oracle_flags(distances,torch.tensor([],dtype=torch.long),True)['oracle_NE'] is None
    assert oracle_flags(distances,torch.tensor([],dtype=torch.long),True)['keep_a0_oracle']


def test_guidance_is_public_and_all_comparisons_share_validity_filter():
    boxes=torch.tensor([[.2,.2,.05,.05],[.8,.8,.05,.05],[.5,.5,.05,.05]])
    scores=torch.tensor([.8,.7,.99]);quality=torch.tensor([1.,1.,.1]);geometry=torch.tensor([-8.,0.,0.])
    native=selection_stages(boxes,scores,quality,geometry)
    guided=selection_stages(boxes,scores,quality,geometry,True)
    assert native['valid900'].tolist()==guided['valid900'].tolist()==[0,1]
    assert int(native['top15'][0])==0 and int(guided['top15'][0])==1


def test_endpoint_ne_bounds_are_valid_without_reconstructing_private_target():
    import numpy as np
    from scripts.diagnose_aerialvg_navigation_failures import distance_bounds,cause_category
    target=np.array([10.,0.]);anchors=np.array([[0.,0.],[10.,5.],[20.,0.]]);radii=np.linalg.norm(anchors-target,axis=-1)
    for point in [np.array([0.,0.]),np.array([7.,4.]),np.array([30.,0.])]:
        low,high=distance_bounds(point,anchors,radii);truth=np.linalg.norm(point-target)
        assert low-1e-6<=truth<=high+1e-6
    assert cause_category((30.,30.),0.,'policy_stop',False)=='定位偏差证据'
    assert cause_category((0.,10.),25.,'horizon',False)=='控制或收敛未完成证据'
    assert cause_category((5.,40.),10.,'horizon',False)=='未知'


def test_empty_proposals_keep_a0_and_do_not_create_missing_key_statistics():
    from scripts.diagnose_aerialvg_candidate_bottlenecks import summarize
    record=dict(a0_correct=True,visible=False,stages={'raw900':dict(count=0,oracle20=False,oracle_NE=None,correctable=False,keep_a0_oracle=True)},fusion={'native/top15':dict(available=False,covered=False,rank_correct=False,updated_correct=True,corrected=False,damaged=False)},first_loss={'native/top15':'validity_filter'})
    result=summarize([record])
    assert result['all']['fusion']['native/top15']['updated_correct']==1
    assert result['a0_wrong']['states']==0
    assert result['all']['stages']['raw900']['oracle_NE_nonempty'] is None
