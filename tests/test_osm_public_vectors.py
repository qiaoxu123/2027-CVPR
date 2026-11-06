import pytest
from graphnav.osm_public_vectors import convert_public_osm

def test_unverified_registration_is_rejected():
    with pytest.raises(ValueError,match='Independent'):convert_public_osm({'features':[]},{'verified':False})

def test_bad_independent_error_is_rejected():
    with pytest.raises(ValueError,match='5m'):convert_public_osm({'features':[]},{'verified':True,'validation_n':3,'validation_errors_m':[1,2,6]})

def test_public_vectors_transformed_and_containment_preserved():
    # Synthetic engineering test only; not a real georegistration measurement.
    r={'verified':True,'validation_n':3,'validation_errors_m':[0,0,0],'matrix':[[1,0,0],[0,1,0]],'projected_crs':'EPSG:3857'}
    g={'features':[{'id':'b','properties':{'building':'yes'},'geometry':{'type':'Polygon','coordinates':[[[0,0],[.001,0],[.001,.001],[0,.001],[0,0]]]}},{'id':'p','properties':{'amenity':'parking'},'geometry':{'type':'Polygon','coordinates':[[[.0001,.0001],[.0002,.0001],[.0002,.0002],[.0001,.0002],[.0001,.0001]]]}}]}
    out=convert_public_osm(g,r)
    assert len(out['features'])==2
    assert any(e['relation']=='contains' and e['subject']=='b' and e['object']=='p' for e in out['relations'])
    assert out['features'][0]['area_m2']>10000
