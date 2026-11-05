"""Tests use synthetic PUBLIC map objects; no target labels, camera or GT."""
import copy
import numpy as np
import pytest
import torch

from graphnav.instruction_relation_graph import (
    compile_relation_graph, normalize_predicate, tensorize_relation_graph,
)
from graphnav.instruction_relation_encoder import (
    InstructionRelationGraphEncoder, batch_relation_graphs,
)


def _span(text, substring):
    start=text.index(substring)
    return [start,start+len(substring)]


def _case(indirect=True, predicate="behind", frame="reference_frame_unknown"):
    if indirect:
        text="A blue car in the parking lot behind One Stop."
        names=("car","parking lot","One Stop")
        sources=(("e0","e1","in","topological"),
                 ("e1","e2",predicate,frame))
        paths=[dict(entities=["e0","e1","e2"],
                    relations=[dict(relation_id="r0",traversal="forward"),
                               dict(relation_id="r1",traversal="forward")],
                    public_names=["One Stop"],hops=2,ambiguity=[])]
    else:
        text="A blue car near One Stop."
        names=("car","One Stop")
        sources=(("e0","e1",predicate,frame),)
        paths=[dict(entities=["e0","e1"],
                    relations=[dict(relation_id="r0",traversal="forward")],
                    public_names=["One Stop"],hops=1,ambiguity=[])]
    nodes=[dict(id=f"e{i}",role="target" if i==0 else "landmark" if i==len(names)-1 else "reference_object",
                text=n,span=_span(text,n),attributes=[])
           for i,n in enumerate(names)]
    anchor=names[-1]
    a,b=_span(text,anchor)
    relations=[dict(id=f"r{i}",subject=s,object=o,predicate=p,
                    relation_kind="spatial_proposal",
                    source_span=[0,len(text)],source_text=text,
                    reference_frame=dict(frame=f),
                    ambiguity=[])
               for i,(s,o,p,f) in enumerate(sources)]
    parsed=dict(original_text=text,map_name="test_map",target_GT_used=False,
                target=dict(entity="e0",category="Car"),entities=nodes,
                relations=relations,
                direct_relations=[] if indirect else paths,
                indirect_relations=paths if indirect else [],
                landmarks=[dict(start=a,end=b,public_ids=["L1"],public_names=[anchor])],
                ambiguity=[])
    objects={"L1":dict(contour=[[20,20],[40,20],[40,50],[20,50]],name="One Stop",object_type="Building")}
    return parsed,objects,[0,0,100,100]


def test_direct_path_keeps_unknown_target_geometry_masked():
    parsed,objects,bounds=_case(indirect=False,predicate="near",frame="metric_or_adjacency")
    graph=compile_relation_graph(parsed,objects,bounds)
    assert graph["summaries"]["direct_paths"]==1
    assert graph["summaries"]["indirect_paths"]==0
    assert graph["paths"][0]["eligible_for_geometric_scoring"]
    assert not graph["nodes"][0]["geometry_valid"]
    assert graph["nodes"][1]["geometry_valid"]
    assert not graph["edges"][0]["geometry_valid"]  # missing GT/candidate => masked
    packed=tensorize_relation_graph(graph)
    assert packed["node_geo_mask"].tolist()[:2]==[False,True]
    assert not packed["edge_geo_mask"][0]
    assert packed["target_index"]==0


def test_indirect_path_keeps_intermediate_unmapped_and_behind_unresolved():
    parsed,objects,bounds=_case()
    graph=compile_relation_graph(parsed,objects,bounds)
    assert graph["summaries"]["indirect_paths"]==1
    assert graph["paths"][0]["nodes"]==["e0","e1","e2"]
    assert graph["paths"][0]["hops"]==2
    assert not graph["nodes"][1]["geometry_valid"]
    assert graph["nodes"][2]["geometry_valid"]
    assert graph["edges"][1]["predicate"]=="behind"
    assert graph["edges"][1]["direction_xy"]==[0.,0.]
    assert not graph["edges"][1]["direction_xy_valid"]
    assert not graph["paths"][0]["eligible_for_geometric_scoring"]


def test_cardinal_direction_not_forced_into_world_frame_without_verification():
    p,objs,b=_case(indirect=False,predicate="north of",frame="world_cardinal")
    unverified=compile_relation_graph(p,objs,b)
    verified=compile_relation_graph(p,objs,b,world_north_verified=True)
    assert not unverified["edges"][0]["direction_xy_valid"]
    assert unverified["edges"][0]["direction_xy"]==[0,0]
    assert "geographic_north_not_verified" in unverified["paths"][0]["ambiguity"]
    assert verified["edges"][0]["direction_xy_valid"]
    assert verified["edges"][0]["direction_xy"]==[0,1]
    assert verified["paths"][0]["eligible_for_geometric_scoring"]


def test_no_gt_and_invalid_or_reversed_path_fails_closed():
    p,objs,b=_case(indirect=False,predicate="near",frame="metric_or_adjacency")
    p["target_GT_used"]=True
    with pytest.raises(ValueError,match="GT-free"):
        compile_relation_graph(p,objs,b)
    p["target_GT_used"]=False
    p["target_world"]=[20,30]
    with pytest.raises(ValueError,match="Private"):
        compile_relation_graph(p,objs,b)
    del p["target_world"]
    p["direct_relations"][0]["relations"][0]["traversal"]="reverse"
    with pytest.raises(ValueError,match="inconsistent"):
        compile_relation_graph(p,objs,b)


def test_no_silent_capacity_truncation_and_unknown_predicate():
    p,objs,b=_case(indirect=True)
    graph=compile_relation_graph(p,objs,b)
    with pytest.raises(ValueError,match="increase capacity"):
        tensorize_relation_graph(graph,max_nodes=2)
    assert normalize_predicate("possessive region association")=="part_of"
    assert normalize_predicate("not an identifiable predicate")=="unknown"


def test_trainable_encoder_masked_geometry_and_gradient():
    p,objs,b=_case(indirect=True)
    graph=compile_relation_graph(p,objs,b)
    batch=batch_relation_graphs([graph])
    torch.manual_seed(7)
    model=InstructionRelationGraphEncoder(text_dim=16,hidden=32,layers=2,dropout=0.)
    node_emb=torch.randn(1,32,16)
    edge_emb=torch.randn(1,64,16)
    model.eval()
    a=model(batch,node_emb,edge_emb)
    assert a["target_present"].tolist()==[True]
    assert a["target_state"].shape==(1,32)
    assert a["edge_states"].shape==(1,64,32)
    # Changing unknown physical coordinates must have NO effect.
    corrupted={k:v.clone() for k,v in batch.items()}
    corrupted["node_geometry"][0,0]=100000.
    corrupted["node_geometry"][0,1]=-100000.
    corrupted["edge_geometry"][:]=100000.
    z=model(corrupted,node_emb,edge_emb)
    assert torch.allclose(a["graph_state"],z["graph_state"],atol=1e-5)
    model.train()
    out=model(batch,node_emb,edge_emb)
    loss=out["graph_state"].square().sum()+out["edge_states"].square().sum()
    loss.backward()
    assert model.node_text[0].weight.grad is not None
    assert torch.isfinite(model.node_text[0].weight.grad).all()
    assert float(model.node_text[0].weight.grad.abs().sum())>0.
