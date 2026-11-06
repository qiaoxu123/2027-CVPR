import numpy as np,pytest
from rasterio.warp import transform
from tools.osm_map.registration import fit_controls

def points():
 xy=np.array([[i%5*40.,i//5*50.] for i in range(15)])
 en=xy+np.array([545000.,258000.]);lon,lat=transform('EPSG:27700','EPSG:4326',en[:,0],en[:,1])
 return [dict(id=str(i),x=float(p[0]),y=float(p[1]),lon=lon[i],lat=lat[i],map_name='cambridge_block_3' if i<12 else 'cambridge_block_4',split='fit' if i<10 else 'validation') for i,p in enumerate(xy)]

def test_independent_validation_and_roundtrip():
 r=fit_controls(points());assert r['verified'];assert r['validation']['max_m']<.1;assert r['validation_n']==5;assert len(r['per_map_validation'])==2

def test_insufficient_points_and_duplicate_are_rejected():
 with pytest.raises(ValueError):fit_controls(points()[:12])
 p=points();p[-1]['x']=p[0]['x'];p[-1]['y']=p[0]['y']
 with pytest.raises(ValueError,match='重复'):fit_controls(p)

def test_wrong_validation_is_not_approved():
 p=points()
 for v in p[10:]:v['lon']+=.001
 assert not fit_controls(p)['verified']

def test_same_osm_point_cannot_be_reused_for_validation():
 p=points();p[-1]['lon']=p[0]['lon'];p[-1]['lat']=p[0]['lat']
 with pytest.raises(ValueError,match='OSM'):fit_controls(p)
