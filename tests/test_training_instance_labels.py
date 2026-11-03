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
