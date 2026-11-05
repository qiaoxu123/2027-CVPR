import torch

from graphnav.instruction_graph_decode import decode_span_graph
from graphnav.instruction_graph_parser_head import EDGE_CLASSES
from graphnav.instruction_relation_graph import ROLES,FRAMES


def _example_output():
    text="A blue car in the parking lot behind One Stop."
    words=["car","parking lot","One Stop"]
    spans=[(text.index(w),text.index(w)+len(w)) for w in words]
    n=3
    roles=torch.full((1,n,len(ROLES)),-9.)
    roles[0,0,ROLES.index("target")]=9.
    roles[0,1,ROLES.index("intermediate")]=9.
    roles[0,2,ROLES.index("public_landmark")]=9.
    edges=torch.full((1,n,n,len(EDGE_CLASSES)),-9.)
    edges[:,:,:,0]=9.  # no edge by default
    for src,dst,name in [(0,1,"inside"),(1,2,"behind")]:
        edges[0,src,dst,0]=-9.
        edges[0,src,dst,EDGE_CLASSES.index(name)]=9.
    frames=torch.full((1,n,n,len(FRAMES)),-9.)
    frames[:,:,:,FRAMES.index("unknown")]=9.
    frames[0,0,1,FRAMES.index("unknown")]=-9.
    frames[0,0,1,FRAMES.index("topological")]=9.
    frames[0,1,2,FRAMES.index("unknown")]=-9.
    frames[0,1,2,FRAMES.index("reference_frame_unknown")]=9.
    mask=torch.ones((1,n,n),dtype=torch.bool)
    mask[:,torch.arange(n),torch.arange(n)]=False
    return text,spans,dict(
        role_logits=roles,edge_logits=edges,frame_logits=frames,
        valid_nodes=torch.ones((1,n),dtype=torch.bool),
        valid_pairs=mask,
    )


def test_decode_neural_predictions_into_indirect_public_anchor_path():
    text,spans,out=_example_output()
    bounds=[0,0,100,100]
    public={"L1":dict(contour=[[20,20],[40,20],[40,50],[20,50]])}
    bindings={2:dict(public_ids=["L1"],public_names=["One Stop"])}
    graph=decode_span_graph(
        out,text,spans,bindings,public,bounds,map_name="test_map",
        min_probability=.5
    )
    assert graph["map_name"]=="test_map"
    assert graph["target_id"]=="e0"
    assert len(graph["paths"])==1
    assert graph["paths"][0]["kind"]=="indirect"
    assert graph["paths"][0]["hops"]==2
    assert not graph["paths"][0]["eligible_for_geometric_scoring"]
    assert graph["nodes"][1]["role"]=="intermediate"
    assert not graph["nodes"][1]["geometry_valid"]
    assert graph["nodes"][2]["geometry_valid"]
    assert graph["edges"][1]["predicate"]=="behind"
    assert graph["edges"][1]["model_confidence"]>.99


def test_decoder_does_not_invent_target_when_role_confidence_low():
    text,spans,out=_example_output()
    out["role_logits"][:,:,:]=0.
    out["role_logits"][:,:,ROLES.index("target")]=-10.
    graph=decode_span_graph(
        out,text,spans,{}, {},[0,0,100,100],
        min_probability=.5
    )
    assert graph["target_id"] is None
    assert not graph["paths"]
    assert "target_role_unresolved" in graph["global_ambiguity"]
