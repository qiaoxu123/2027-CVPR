import numpy as np
from graphnav.training_instance_labels import instance_core_labels


def test_other_vehicle_inside20m_is_not_a_target_positive():
    points=np.array([[0.,0.],[3.,0.],[1.5,0.],[10.,10.]])
    result=instance_core_labels(points,np.ones(4,bool),np.array([[0.,0.],[3.,0.]]),[7,8],7)
    np.testing.assert_array_equal(result,[1,0,-1,-1])


def test_overlapping_or_invalid_observations_remain_unknown():
    points=np.array([[0.,0.],[0.,0.]])
    result=instance_core_labels(points,np.array([True,False]),np.array([[0.,0.],[0.,0.]]),[7,8],7)
    np.testing.assert_array_equal(result,[-1,-1])


def test_private_instance_labels_are_not_part_of_model_forward_contract():
    import inspect
    from graphnav.visual_chain_position import WorldLikelihoodPositionHead
    from scripts.train_local_chain_position import INPUTS
    private={'target_world','target_id','object_ids','instance_labels','physical_group'}
    assert not (private&set(INPUTS))
    assert not (private&set(inspect.signature(WorldLikelihoodPositionHead.forward).parameters))


def test_conservative_contour_background_keeps_boundary_and_conflicts_unknown():
    from graphnav.training_instance_labels import conservative_car_background
    points=np.array([[0.,0.],[2.,0.],[5.,0.],[5.,0.],[0.,0.]])
    core=np.array([1,-1,-1,1,1],dtype=np.int8);valid=np.array([True,True,True,True,False])
    contour=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    result=conservative_car_background(core,points,valid,contour,2.)
    np.testing.assert_array_equal(result,[1,-1,0,-1,-1])


def test_invalid_contour_is_abstention_not_a_fabricated_mask():
    from graphnav.training_instance_labels import conservative_car_background
    result=conservative_car_background(np.array([1],np.int8),np.array([[0.,0.]]),np.array([True]),np.array([[0.,0.],[1.,1.],[0.,1.],[1.,0.]]),2.)
    np.testing.assert_array_equal(result,[-1])


def test_background_statistics_do_not_claim_other_vehicle_instances():
    from graphnav.training_instance_labels import instance_annotation_counts
    result=instance_annotation_counts(np.array([1,-1,-1],np.int8),np.array([1,0,0],np.int8))
    assert result['frames_with_target_and_other_car_core']==0
    assert result['eligible_loss_frames']==1 and result['background_negative_points']==2
    assert result['frames_with_target_core']==1 and result['retained_target_core_frames']==1
