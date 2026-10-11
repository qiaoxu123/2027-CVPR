import inspect
import numpy as np
import pytest
from experiments.active_view_selection.planner import View,legal,library,select,history_coverage,modes
from experiments.multiview_rgb_grounding.render import project

def fixture():
    p=dict(bounds=[0,0,200,200],grid_xy=[[.5,.5],[.6,.5],[.9,.5]],static_logits=[3,2,1],pose_world=[50,100,50,0])
    c=dict(xyz=[50,100,50],uav_yaw=0,camera_yaw=0,camera_tilt=0,ground=0,instruction='a building')
    return p,c

def test_independent_camera_yaw_does_not_move_uav():
    v=View((10,20,50),np.pi/2,40,'hold',0)
    assert v.xyz==(10,20,50) and v.uav_yaw==0
    assert v.camera_pose()==[10,20,50,np.pi/2]
    points=np.array([[10,40,0.]])
    a=project(points,v.camera_pose(),40)[0];b=project(points,[10,20,50,0],40)[0]
    assert not np.allclose(a,b)

def test_legal_caps_and_invalid_terrain_proxy():
    assert legal([50,50,50],[100,50,60],[0,0,200,200],0,[])
    for xyz in [[100.1,50,50],[50,50,61],[50,50,9],[50,50,101],[-1,50,50]]:
        assert not legal([50,50,50],xyz,[0,0,200,200],0,[])

def test_public_obstacle_excludes_crossing():
    box=dict(type='Building',roof=45,polygon=np.array([[60,40],[80,40],[80,60],[60,60]]))
    assert not legal([50,50,40],[100,50,40],[0,0,200,200],0,[box])
    assert legal([50,50,50],[100,50,50],[0,0,200,200],0,[box])

def test_grid_world_and_merge():
    p,c=fixture();g,ids,w=modes(p,np.zeros(3))
    assert len(g)==2 and g[0]['ids']==[0,1]
    assert 100<=g[0]['center'][0]<=120 and np.isclose(w.sum(),1)

def test_future_history_rejected():
    p,c=fixture()
    with pytest.raises(ValueError,match='future'):
        history_coverage(p,[dict(source_step=6,credible=True,center_world=[100,100])],5)
    x=history_coverage(p,[dict(source_step=5,credible=True,center_world=[100,100])],5)
    assert np.array_equal(x,[1,0,0])

def test_budget_and_legal_library():
    p,c=fixture();views=library(p,c,0,[],np.zeros(3))
    assert {v.camera_tilt for v in views}=={0,20,40,60}
    assert all(legal(c['xyz'],v.xyz,p['bounds'],0,[]) for v in views)
    with pytest.raises(ValueError,match='budget'):
        select('active_view',p,c,0,[],np.zeros(3),2,np.random.default_rng(0))

def test_camera_only_fixed_and_nadir_constraints():
    p,c=fixture()
    for name in ['fixed_nadir','camera_only','move_nadir']:
        v=select(name,p,c,0,[],np.zeros(3),0,np.random.default_rng(0))
        if name!='move_nadir':assert np.allclose(v.xyz,c['xyz'])
        if name!='camera_only':assert v.camera_tilt==0 and v.camera_yaw==c['uav_yaw']

def test_random_reproducible():
    p,c=fixture()
    a=select('random_legal',p,c,0,[],np.zeros(3),0,np.random.default_rng(42))
    b=select('random_legal',p,c,0,[],np.zeros(3),0,np.random.default_rng(42))
    assert a==b

def test_target_and_cloud_isolation():
    import experiments.active_view_selection.planner as planner
    p,c=fixture();before=select('active_view',p,c,0,[],np.zeros(3),0,np.random.default_rng(0))
    # Mutating external sealed labels/RGB cannot affect arguments or output.
    sealed={'target_object_id':'999','rgb':np.zeros((2,2,3))};sealed['target_object_id']='7';sealed['rgb'][:]=255
    after=select('active_view',p,c,0,[],np.zeros(3),0,np.random.default_rng(0));assert before==after
    source=inspect.getsource(planner)
    assert all(s not in source for s in ['read_cloud(', 'target_object_id', 'torch.load(', 'cv2.imread(', 'exposure('])

def test_visible_only_association_matches_full_cloud():
    from experiments.active_view_selection.run import visible_regions
    from experiments.multiview_rgb_grounding.render import associate,regions
    xyz=np.array([[0,0,0],[1,1,0],[2,2,0],[10,10,0],[2,1,0],[100,100,0]],float)
    objects={'0':dict(contour=[[-1,-1],[3,-1],[3,3],[-1,3]],position=[1,1,0],dimension=[4,4,2],object_type='Building'),'1':dict(contour=[[1.5,.5],[3,.5],[3,3],[1.5,3]],position=[2,2,0],dimension=[4,4,2],object_type='Building')}
    win=np.array([[0,1,1],[2,4,-1],[3,3,1]]);frame={'winner':win}
    gate=dict(instance_min_native_side_pixels=1,instance_visible_measured_pixels_min=1,instance_bbox_valid_fraction_min=0)
    assignment,names,_,_=associate(xyz,objects);a,am=regions(frame,assignment,names,objects,gate);b,bm=visible_regions(frame,xyz,objects,gate)
    assert a==b and np.array_equal(am,bm)

def test_exact_frustum_culling_keeps_original_render():
    from experiments.active_view_selection.sensor import render_culled
    from experiments.multiview_rgb_grounding.render import render
    rng=np.random.default_rng(20);xyz=rng.uniform([-300,-300,-5],[300,300,80],size=(4000,3)).astype(np.float32);rgb=rng.integers(0,256,size=(4000,3),dtype=np.uint8)
    # Include exact depth ties; sorted original indices must remain tie breaker.
    xyz[1]=xyz[0]
    for yaw in [0,.7]:
        for tilt in [0,20,40,60]:
            p=[0,0,100,yaw];a=render(xyz,rgb,p,tilt,radius=1);b=render_culled(xyz,rgb,p,tilt)
            for key in ['rgb','winner','valid']:assert np.array_equal(a[key],b[key])
            assert np.array_equal(a['depth'],b['depth'],equal_nan=True)
