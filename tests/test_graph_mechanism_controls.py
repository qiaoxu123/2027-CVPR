import torch
import pytest
from graphnav.dynamic_runtime import graph_control_inputs
from graphnav.dynamic_runtime import visual_control_vectors
from types import SimpleNamespace


def test_graph_controls_preserve_a3_inputs_and_invalid_nodes():
    original=dict(node_valid=torch.tensor([[True,True,False]]),
        adjacency=torch.ones(1,3,3,dtype=torch.bool),edges=torch.ones(1,3,3,4),base_logits=torch.ones(1,784))
    self_only=graph_control_inputs(original,'self')
    assert torch.equal(self_only['adjacency'][0],torch.diag(torch.tensor([True,True,False])))
    assert original['adjacency'].all()
    zero=graph_control_inputs(original,'zero-edge-features')
    assert zero['edges'].count_nonzero()==0 and original['edges'].all()
    assert zero['base_logits'] is original['base_logits']
    assert graph_control_inputs(original,'full') is original
    with pytest.raises(ValueError):graph_control_inputs(original,'invented-topology')


def test_visual_controls_preserve_geometry_and_shuffle_only_actual_same_type_vectors():
    rois=[SimpleNamespace(object_type=x) for x in ('Building','TrafficRoad','Building')]
    vectors=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    shuffled=visual_control_vectors(rois,vectors,'same-type-shuffle')
    assert torch.equal(shuffled,vectors[[2,1,0]])
    assert torch.equal(vectors,torch.tensor([[1.,2.],[3.,4.],[5.,6.]]))
    assert visual_control_vectors(rois,vectors,'zero-vectors').count_nonzero()==0
    assert visual_control_vectors(rois,vectors,'full') is vectors
