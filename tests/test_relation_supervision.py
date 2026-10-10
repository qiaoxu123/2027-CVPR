import numpy as np
from graphnav.semantic_geometry import SpatialRelation
from graphnav.relation_supervision import legal_region


def entity(kind='building'):
    return {'kind':kind,'geometry_m':np.array([[-2,-2],[-2,2],[2,2],[2,-2]],dtype=np.float32)}


def test_cardinals_are_world_xy_and_keep_multiple_legal_cells():
    points=np.array([[10,0],[20,20],[-10,0],[0,10]])
    e=legal_region(points,[entity()],SpatialRelation('east',((0,),)))
    w=legal_region(points,[entity()],SpatialRelation('west',((0,),)))
    assert e.tolist()==[True,True,False,False]
    assert w.tolist()==[False,False,True,False]


def test_ambiguous_binding_and_surface_along_are_skipped():
    points=np.array([[10,0]])
    assert legal_region(points,[entity(),entity()],SpatialRelation('east',((0,1),))) is None
    assert legal_region(points,[entity('road_surface')],SpatialRelation('along',((0,),))) is None
    assert legal_region(points,[entity()],SpatialRelation('left',((0,),))) is None


def test_between_does_not_extend_beyond_anchors():
    other=entity();other['geometry_m']=other['geometry_m']+np.array([100,0])
    mask=legal_region(np.array([[50,0],[110,0],[50,30]]),[entity(),other],SpatialRelation('between',((0,),(1,))))
    assert mask.tolist()==[True,False,False]


def test_real_direction_counter_does_not_change_goal_label(tmp_path):
    import json,torch
    from graphnav.relation_supervision import RelationDataset,collate_relation
    row={'map_name':'test','instruction':'Find the target to the east of Tower',
         'world_bounds':[0,0,100,100],'physical_map_sizes':[100,100],'target_xy':[.8,.5],
         'landmarks':[{'name':'Tower','contour':[[.4,.4],[.6,.4],[.6,.6],[.4,.6]]}]}
    p=tmp_path/'train.jsonl';p.write_text(json.dumps(row)+'\n')
    data=RelationDataset(p,max_roads=0)
    x=data[0]
    assert x['counter_eligible'] and x['relation_eligible']
    assert x['counter_text']=='Find the target to the west of Tower'
    assert torch.equal(x['target_xy'],torch.tensor([.8,.5]))
    assert not (x['counter_allowed'] & x['relation_allowed']).any()
    batch=collate_relation([x]);assert batch['counter_text']==[x['counter_text']]


def test_short_world_direction_phrase_is_swapped_without_rewriting_names(tmp_path):
    import json
    from graphnav.relation_supervision import RelationDataset
    row={'map_name':'test','instruction':'Find the building east of West Tower',
         'world_bounds':[0,0,100,100],'physical_map_sizes':[100,100],'target_xy':[.8,.5],
         'landmarks':[{'name':'West Tower','contour':[[.4,.4],[.6,.4],[.6,.6],[.4,.6]]}]}
    p=tmp_path/'train.jsonl';p.write_text(json.dumps(row)+'\n')
    data=RelationDataset(p,max_roads=0)
    assert data[0]['counter_eligible']
    assert data[0]['counter_text']=='Find the building west of West Tower'
    assert data[0] is data[0]


def test_direction_region_excludes_contour_interior_and_uncertain_edge():
    mask=legal_region(np.array([[1,0],[4,0],[10,0]]),[entity()],SpatialRelation('east',((0,),)))
    assert mask.tolist()==[False,False,True]
