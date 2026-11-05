"""CPU: weak graph safety, public Polygon binding, node permutation, gradients."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from graphnav.instruction_relation_graph import tensorize_relation_graph
from graphnav.unified_instruction_graph_belief import UnifiedInstructionGraphBelief
from graphnav.unified_instruction_graph_data import (
    empty_graph,graph_input_batch,read_graphs,row_signature,
)


def _graph():
    return dict(
        schema="citynav-instruction-relation-graph-v1",
        target_GT_used=False,world_north_verified=False,
        target_id="t",
        nodes=[
            dict(id="t",role="target",phrase="car",category="Car",
                 attributes=[dict(kind="color",value="blue")],
                 geometry=[0.]*8,geometry_valid=False,public_ids=[]),
            dict(id="m",role="intermediate",phrase="parking lot",
                 attributes=[],geometry=[0.]*8,geometry_valid=False,
                 public_ids=[]),
            dict(id="l",role="public_landmark",phrase="One Stop",
                 attributes=[],public_names=["One Stop"],public_ids=["123"],
                 geometry=[.4,.3,.1,.1,.1,.1,1.,1.],geometry_valid=True),
        ],
        edges=[
            dict(id="a",subject="t",object="m",predicate="inside",
                 source_predicate="inside",frame="topological",
                 geometry=[0.]*9,geometry_valid=False,
                 direction_xy=[0.,0.],direction_xy_valid=False),
            dict(id="b",subject="m",object="l",predicate="behind",
                 source_predicate="behind",frame="reference_frame_unknown",
                 geometry=[0.]*9,geometry_valid=False,
                 direction_xy=[0.,0.],direction_xy_valid=False),
        ],
        paths=[dict(kind="indirect",hops=2,
                    nodes=["t","m","l"],edges=[],ambiguity=[])]
    )


class MockBert:
    def encode(self,phrases):
        outputs=[]
        for phrase in phrases:
            seed=int.from_bytes(hashlib.sha256(phrase.encode()).digest()[:4],"big")
            rng=np.random.default_rng(seed)
            outputs.append(torch.tensor(rng.standard_normal(768),dtype=torch.float32))
        return torch.stack(outputs)


def _batch(graphs,records=None):
    if records is None:
        records=[dict(map_name="cambridge_block_1",
                      instruction="Blue car in parking lot behind One Stop",
                      target_xy=[.8,.8],
                      landmarks=[dict(id="123",name="One Stop",
                                      contour=[[.1,.1],[.2,.1],[.2,.2],[.1,.2]])])
                 for _ in graphs]
    return graph_input_batch(graphs,records,MockBert(),"cpu",
                             entity_capacity=2)


def _model_input(batch_size,queries=14):
    torch.manual_seed(42)
    return dict(
        base_logits=torch.randn(batch_size,queries),
        dense_context=torch.randn(batch_size,queries,128),
        pair_geometry=torch.randn(batch_size,queries,2,10),
    )


def test_only_path_terminal_public_id_is_candidate_grounding():
    good=_graph()
    row=dict(instruction="nonsense",map_name="map",target_xy=[.15,.15],
             landmarks=[dict(id="123"),dict(id="unrelated")])
    b=_batch([good],[row])
    assert b["node_anchor"].shape==(1,32,2)
    assert b["node_anchor"][0,2].tolist()==[1.,0.]
    assert b["node_anchor"][0,:2].sum()==0
    assert b["node_text"].shape==(1,32,768)
    assert b["edge_text"].shape==(1,64,768)
    # The target node must preserve visual attributes; not only "car".
    assert torch.equal(b["node_text"][0,0],MockBert().encode(["Car; blue"])[0])


def test_zero_initialized_graph_head_exactly_preserves_full_text_baseline():
    graphs=[_graph(),empty_graph()]
    b=_batch(graphs)
    x=_model_input(2)
    model=UnifiedInstructionGraphBelief(hidden=128,layers=2)
    out=model(packed=b["packed"],node_text=b["node_text"],
              edge_text=b["edge_text"],node_anchor=b["node_anchor"],
              return_diagnostics=True,**x)
    assert torch.equal(out["logits"],x["base_logits"])
    assert torch.equal(out["delta"],torch.zeros_like(x["base_logits"]))
    assert out["supported"].tolist()==[True,False]
    assert out["anchor_weights"][1].abs().max()==0
    assert out["anchor_weights"][0,:,2].sum()>0


def test_learned_graph_head_is_bounded_and_keeps_no_path_untouched():
    b=_batch([_graph(),empty_graph()])
    x=_model_input(2)
    model=UnifiedInstructionGraphBelief(hidden=128,layers=2)
    with torch.no_grad():
        model.anchor_score.weight.normal_(0,.05)
    out=model(packed=b["packed"],node_text=b["node_text"],
              edge_text=b["edge_text"],node_anchor=b["node_anchor"],**x)
    assert (out["delta"].abs()<=1.00001).all()
    assert torch.equal(out["logits"][1],x["base_logits"][1])
    assert out["delta"][0].abs().max()>0.
    loss=torch.nn.functional.cross_entropy(out["logits"][0:1],
                                            torch.tensor([3]))
    loss.backward()
    assert model.anchor_score.weight.grad.abs().sum()>0
    assert model.encoder.forward_message[0][0].weight.grad is not None


def test_node_reindexing_does_not_change_candidate_belief():
    graph=_graph()
    perm=[2,0,1]
    reordered=copy.deepcopy(graph)
    reordered["nodes"]=[reordered["nodes"][i] for i in perm]
    b1=_batch([graph])
    b2=_batch([reordered])
    x=_model_input(1)
    model=UnifiedInstructionGraphBelief(hidden=128,layers=2).eval()
    with torch.no_grad():
        model.anchor_score.weight.normal_(0,.03)
        a=model(packed=b1["packed"],node_text=b1["node_text"],
                edge_text=b1["edge_text"],node_anchor=b1["node_anchor"],**x)
        b=model(packed=b2["packed"],node_text=b2["node_text"],
                edge_text=b2["edge_text"],node_anchor=b2["node_anchor"],**x)
    torch.testing.assert_close(a["logits"],b["logits"],atol=2e-6,rtol=2e-6)


def test_unsafe_identity_fails_and_missing_gt_is_never_a_model_kwarg():
    graph=_graph();row=dict(
        map_name="x",instruction="y",landmarks=[dict(id="123")],
        target_xy=[.25,.25],
    )
    b=_batch([graph],[row])
    model=UnifiedInstructionGraphBelief()
    with pytest.raises(TypeError):
        model(packed=b["packed"],node_text=b["node_text"],
              edge_text=b["edge_text"],node_anchor=b["node_anchor"],
              target_xy=torch.tensor([[.25,.25]]),**_model_input(1))
    graph["target_GT_used"]=True
    with pytest.raises(ValueError,match="GT-free"):
        _batch([graph],[row])


def test_aligned_graph_file_rejects_changed_instruction(tmp_path):
    row=dict(map_name="map",instruction="blue car",target_xy=[.5,.5])
    p=tmp_path/"weak.jsonl"
    p.write_text(json.dumps(dict(index=0,map_name="map",
                                 row_signature=row_signature(row),
                                 graph=empty_graph()))+"\n")
    assert read_graphs(p,[row])[0]["target_id"] is None
    changed=dict(row,instruction="red car")
    with pytest.raises(ValueError,match="alignment"):
        read_graphs(p,[changed])
