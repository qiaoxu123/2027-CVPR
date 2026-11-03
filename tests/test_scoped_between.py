from graphnav.scoped_between import scoped_between


def fixture():
    nodes={i:dict(phrase=x) for i,x in enumerate(['car','parking lot','North Road','Hall'])}
    graph=dict(target_token=0,scope_uncertain=False,edges=[dict(a=0,b=1,relation='in'),dict(a=1,b=2,relation='between'),dict(a=1,b=3,relation='between')])
    return nodes,graph,{2:['road1','road2'],3:['hall']}


def test_between_keeps_intermediate_scope_and_multi_polygon_alternatives():
    n,g,p=fixture();r=scoped_between(n,g,p)
    assert len(r)==1 and r[0]['target_to_owner']==[0,1]
    assert r[0]['kind']=='two-public-anchors'
    assert r[0]['endpoints'][0]['public_ids']==['road1','road2']


def test_near_region_not_silently_treated_as_target_containment():
    n,g,p=fixture();g['edges'][0]['relation']='near'
    assert scoped_between(n,g,p)==[]


def test_quantity_two_is_not_two_fabricated_references():
    n,g,p=fixture();g['edges'].pop()
    assert scoped_between(n,g,p)==[]


def test_unknown_scope_or_three_endpoints_abstains():
    n,g,p=fixture();g['scope_uncertain']=True;assert scoped_between(n,g,p)==[]
    g['scope_uncertain']=False;n[4]=dict(phrase='extra');g['edges'].append(dict(a=1,b=4,relation='between'))
    assert scoped_between(n,g,p)==[]


def test_unbound_proper_name_is_not_a_visual_instance_label():
    n,g,p=fixture();p.pop(3);n[3]['proper_name_hint']=True
    r=scoped_between(n,g,p)
    assert r[0]['kind']=='unresolved-name-endpoint'
    assert r[0]['endpoints'][1]['name_unresolved']
