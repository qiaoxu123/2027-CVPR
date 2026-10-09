import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from graphnav.dynamic_runtime import NativeObservationBuilder


def test_native_boundary_avoids_unused_sbf_canvas_and_preserves_current_pose(monkeypatch):
    mapdata=ModuleType('sbfnav.mapdata');mapdata.GROUND_LEVEL={'map':10.}
    monkeypatch.setitem(sys.modules,'sbfnav.mapdata',mapdata)
    poses=np.array([[1,2,30,0],[3,4,25,.5]],dtype=np.float64)
    class Renderer:
        def current_rgbd(self,name,pose,ground):
            assert name=='map' and ground==10 and np.array_equal(pose,poses[-1])
            return dict(bgr=np.ones((384,384,3),dtype=np.uint8),surface_height_m=np.ones((384,384)),pose=pose.copy())
    class Builder:
        def build_from_poses(self,*args):raise AssertionError('unused SBF canvas must not be built')
    runtime=NativeObservationBuilder.__new__(NativeObservationBuilder)
    runtime.builder=Builder();runtime.renderer=Renderer()
    result=runtime.build_from_poses(SimpleNamespace(map_name='map'),poses)
    assert np.array_equal(result.pose,poses[-1]) and result.step==1
    assert 'surface_height_m' not in result.current_observation
    result.pose[0]=999
    assert poses[-1,0]==3
    with pytest.raises(ValueError):runtime.build_from_poses(SimpleNamespace(map_name='map'),[])
