from graphnav.region_chain_requirement import region_requirement


def test_car_inside_row_inside_parking_infers_region_without_private_class():
    nodes={0:dict(lemma='car',phrase='blue car'),1:dict(lemma='row',phrase='third row'),2:dict(lemma='lot',phrase='the parking lot')}
    graph=dict(target_token=0,scope_uncertain=False,edges=[dict(a=0,b=1,relation='in'),dict(a=1,b=2,relation='in')])
    r=region_requirement(nodes,graph)
    assert r['region']=='Parking' and r['path']==[0,1,2]


def test_nearby_parking_and_reference_containment_do_not_imply_target_inside():
    nodes={0:dict(lemma='car',phrase='red car'),1:dict(lemma='building',phrase='house'),2:dict(lemma='lot',phrase='parking lot')}
    graph=dict(target_token=0,scope_uncertain=False,edges=[dict(a=0,b=1,relation='near'),dict(a=1,b=2,relation='in')])
    assert region_requirement(nodes,graph) is None


def test_building_target_not_replaced_by_its_adjacent_parking_reference():
    nodes={0:dict(lemma='building',phrase='white building'),1:dict(lemma='lot',phrase='parking lot')}
    graph=dict(target_token=0,scope_uncertain=False,edges=[dict(a=0,b=1,relation='beside')])
    assert region_requirement(nodes,graph)['region']=='Building'


def test_uncertain_car_scope_abstains_and_floor_is_not_ground_region():
    nodes={0:dict(lemma='car',phrase='car'),1:dict(lemma='lot',phrase='parking lot')}
    graph=dict(target_token=0,scope_uncertain=True,edges=[dict(a=0,b=1,relation='in')])
    assert region_requirement(nodes,graph) is None
    assert region_requirement({0:dict(lemma='ground',phrase='ground floor')},dict(target_token=0,edges=[])) is None


def test_row_membership_can_bridge_to_containing_parking_but_other_of_edges_cannot():
    nodes={0:dict(lemma='car',phrase='car'),1:dict(lemma='row',phrase='row'),2:dict(lemma='car',phrase='cars'),3:dict(lemma='lot',phrase='parking lot')}
    graph=dict(target_token=0,scope_uncertain=False,edges=[dict(a=0,b=1,relation='in'),dict(a=1,b=2,relation='of'),dict(a=2,b=3,relation='in')])
    assert region_requirement(nodes,graph)['region']=='Parking'
    nodes[1]=dict(lemma='roof',phrase='roof')
    assert region_requirement(nodes,graph) is None


def test_parking_grounds_is_parking_not_generic_ground():
    nodes={0:dict(lemma='ground',phrase='The parking grounds')}
    assert region_requirement(nodes,dict(target_token=0,edges=[]))['region']=='Parking'
