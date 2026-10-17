import torch
from graphnav.geombelief_inputs import SPATIAL,mask_geometry,shuffle_manifest

def inputs():
    x={k:torch.randn(2,4,10) for k in SPATIAL}
    x['pair_geometry']=torch.randn(2,784,4,10)
    x['grid_xy']=torch.rand(2,784,2)
    x['node_types']=torch.ones(2,4,dtype=torch.long)
    return x

def test_name_only_removes_all_spatial_bypasses():
    a=inputs();b=inputs()
    b['grid_xy']=a['grid_xy'];b['node_types']=a['node_types']
    x=mask_geometry(a,'name_type');y=mask_geometry(b,'name_type')
    for k in SPATIAL:assert not x[k].any() and torch.equal(x[k],y[k])
    assert x['grid_xy'] is a['grid_xy']
    assert a['nodes'].any()

def test_simple_channels_no_signed_boundary_or_xy_magnitude():
    a=inputs()
    for arm,channels in [('distance',[2]),('direction',[3,4]),('distance_direction',[2,3,4])]:
        x=mask_geometry(a,arm)
        for c in range(10):
            assert torch.equal(x['pair_geometry'][...,c],a['pair_geometry'][...,c]) if c in channels else not x['pair_geometry'][...,c].any()
        for k in SPATIAL:
            if k!='pair_geometry':assert not x[k].any()

def test_shuffle_is_reproducible_partitioned_bijection():
    meta=[dict(query_id=str(i),map_name='map') for i in range(7)]
    typ=torch.zeros(7,4,dtype=torch.long);valid=torch.ones(7,4,dtype=torch.bool)
    part=['fit']*4+['dev']*3
    a=shuffle_manifest(meta,typ,valid,valid,part)
    assert a==shuffle_manifest(meta,typ,valid,valid,part)
    donors=[r['donor_index'] for r in a]
    assert sorted(donors)==list(range(7))
    for i,j in enumerate(donors):assert i!=j and part[i]==part[j]
    z=torch.randn(7,784,4,10)
    assert torch.equal(z.sum(0),z[donors].sum(0)) or torch.allclose(z.sum(0),z[donors].sum(0),atol=1e-5)

def test_singleton_is_explicit_not_fabricated():
    a=shuffle_manifest([dict(query_id='x',map_name='m')],torch.zeros(1,1),torch.ones(1,1),torch.ones(1,1),['fit'])
    assert a[0]['unchanged']

def test_actual_network_name_only_invariant_to_all_geometry():
    from graphnav.full_localization import FullTokenGeometryLocalizer
    torch.manual_seed(0)
    model=FullTokenGeometryLocalizer(hidden=16,text_dim=12,heads=2,max_relations=2,dropout=0).eval()
    a=dict(nodes=torch.randn(1,3,10),node_types=torch.zeros(1,3,dtype=torch.long),node_valid=torch.ones(1,3,dtype=torch.bool),entity_valid=torch.ones(1,3,dtype=torch.bool),prior=torch.randn(1,9),prior_buildings=torch.randn(1,9),rules=torch.randn(1,9),relation_fields=torch.randn(1,9,2),pair_geometry=torch.randn(1,9,3,10),grid_xy=torch.rand(1,9,2),edges=torch.randn(1,3,3,19),adjacency=torch.ones(1,3,3,dtype=torch.bool))
    b={k:torch.randn_like(v) if v.dtype.is_floating_point and k!='grid_xy' else v for k,v in a.items()}
    tok=torch.randn(1,5,12);names=torch.randn(1,3,12);mask=torch.ones(1,5,dtype=torch.bool)
    with torch.no_grad():
        x=model(mask_geometry(a,'name_type'),variant='geometry',tokens=tok,token_mask=mask,node_emb=names)
        y=model(mask_geometry(b,'name_type'),variant='geometry',tokens=tok,token_mask=mask,node_emb=names)
    assert torch.equal(x.belief_logits,y.belief_logits)
