import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from graphnav.renderer import CityNavObservedViewRenderer


def test_rgbd_current_pose_exactly_matches_existing_native_rgb(tmp_path):
    size=96
    height=np.full((size,size),3,np.float32)
    height[30:40,30:40]=-1
    with rasterio.open(tmp_path/'map.tif','w',driver='GTiff',height=size,width=size,
                       count=1,dtype='float32',transform=from_origin(0,size,1,1)) as f:f.write(height,1)
    rgb=np.zeros((size,size,3),np.uint8);rgb[...,0]=np.arange(size)[None,:];rgb[...,1]=np.arange(size)[:,None];rgb[...,2]=120
    cv2.imwrite(str(tmp_path/'map.png'),rgb)
    renderer=CityNavObservedViewRenderer(tmp_path,tmp_path/'output',image_size=384)
    pose=[48,48,20,.37]
    observation=renderer.current_rgbd('map',pose,0)
    old=renderer.render('map','episode',0,[*pose,0],0)
    assert np.array_equal(cv2.imread(old['path']),observation['bgr'])
    assert observation['depth_m'].shape==(384,384)
    valid=observation['depth_valid']
    # Away from interpolation borders, source height is exactly 3m.
    core=valid & (np.abs(observation['surface_height_m']-3)<1e-5)
    np.testing.assert_allclose(observation['depth_m'][core],17,atol=1e-5)
    assert (observation['depth_m'][~valid]==0).all()
    np.testing.assert_allclose(observation['footprint_world'],old['footprint_world'])
    cached_height=renderer._height;cached_valid=renderer._height_valid
    repeated=renderer.current_rgbd('map',pose,0)
    assert renderer._height is cached_height and renderer._height_valid is cached_valid
    for key in ('bgr','depth_m','depth_valid','surface_height_m'):
        assert np.array_equal(observation[key],repeated[key])
