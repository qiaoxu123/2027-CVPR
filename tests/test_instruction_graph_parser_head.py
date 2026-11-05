import torch
import pytest

from graphnav.instruction_relation_graph import ROLES,PREDICATES,FRAMES
from graphnav.instruction_graph_parser_head import (
    EDGE_CLASSES,InstructionSpanGraphParser,supervised_parser_loss
)


def test_span_parser_predicts_directed_pairs_and_gradients():
    torch.manual_seed(11)
    module=InstructionSpanGraphParser(text_dim=24,hidden=32,dropout=0.)
    token_hidden=torch.randn(2,9,24)
    span_mask=torch.zeros((2,3,9),dtype=torch.bool)
    span_mask[:,0,1:3]=True
    span_mask[:,1,3:5]=True
    span_mask[:,2,5:7]=True
    valid=torch.tensor([[True,True,True],[True,True,False]])
    output=module(token_hidden,span_mask,valid)
    assert output["role_logits"].shape==(2,3,len(ROLES))
    assert output["edge_logits"].shape==(2,3,3,len(EDGE_CLASSES))
    assert output["frame_logits"].shape==(2,3,3,len(FRAMES))
    assert not output["valid_pairs"][0,0,0]
    assert not output["valid_pairs"][1,0,2]
    roles=torch.tensor([[ROLES.index("target"),ROLES.index("intermediate"),
                         ROLES.index("public_landmark")],
                        [ROLES.index("target"),ROLES.index("public_landmark"),-100]])
    edges=torch.full((2,3,3),-100)
    frames=torch.full((2,3,3),-100)
    for batch,n in [(0,3),(1,2)]:
        for src in range(n):
            for dst in range(n):
                if src!=dst:edges[batch,src,dst]=0
    edges[0,0,1]=PREDICATES.index("inside")+1
    edges[0,1,2]=PREDICATES.index("behind")+1
    edges[1,0,1]=PREDICATES.index("near")+1
    frames[0,0,1]=FRAMES.index("topological")
    frames[0,1,2]=FRAMES.index("reference_frame_unknown")
    frames[1,0,1]=FRAMES.index("metric_or_adjacency")
    parts=supervised_parser_loss(output,roles,edges,frames)
    assert torch.isfinite(parts["total"])
    parts["total"].backward()
    assert module.phrase[0].weight.grad is not None
    assert float(module.phrase[0].weight.grad.abs().sum())>0.


def test_parser_rejects_missing_span_tokens_and_shapes():
    net=InstructionSpanGraphParser(text_dim=8,hidden=16)
    hidden=torch.randn(1,4,8)
    spans=torch.zeros((1,2,4),dtype=torch.bool)
    valid=torch.tensor([[True,False]])
    with pytest.raises(ValueError,match="zero aligned"):
        net(hidden,spans,valid)
