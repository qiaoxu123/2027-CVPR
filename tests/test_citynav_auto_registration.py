import inspect,json
import numpy as np,pytest
from rasterio.warp import transform
from tools.osm_map.auto_registration import initialize,metrics,estimate

def points():
 xy=np.array([[i%4*30.,i//4*40.] for i in range(12)]);en=xy+[545000.,258000.];lon,lat=transform('EPSG:27700','EPSG:4326',en[:,0],en[:,1]);return [dict(id=str(i),map_name='cambridge_block_3',x=float(v[0]),y=float(v[1]),lon=lon[i],lat=lat[i],split='fit') for i,v in enumerate(xy)]

def test_fit_only_initialization():
 p=points();M,inside=initialize(p);assert inside.sum()==12;assert metrics(M,p)['rmse_m']<.1
 assert 'validation' not in inspect.signature(estimate).parameters

def test_validation_is_rejected_before_io(tmp_path):
 p=points();p[0]['split']='validation'
 with pytest.raises(ValueError,match='Validation'):estimate(tmp_path,'cambridge_block_3',p,tmp_path)

def test_validation_scoring_cannot_change_prediction():
 p=points();M,_=initialize(p);before=M.copy();val=points()
 for x in val:x['lon']+=.001
 assert metrics(M,val)['rmse_m']>20
 np.testing.assert_array_equal(M,before)
