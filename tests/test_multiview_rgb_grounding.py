import numpy as np
from experiments.multiview_rgb_grounding.render import basis, project, render, associate, regions

def test_multiview_camera_basis_and_nadir_projection():
    for tilt in [0,20,40,60]:
        b=basis(.7,tilt);np.testing.assert_allclose(b@b.T,np.eye(3),atol=1e-12)
        assert np.isclose(np.linalg.det(b),1)
    uv,z,good=project([[0,0,0],[1,0,0],[0,1,0]],[0,0,10,0],0,width=100)
    np.testing.assert_allclose(uv,[[50,50],[50,45],[45,50]],atol=1e-10)
    assert good.all() and (z>0).all()

def test_multiview_oblique_is_true_perspective():
    pose=[3,4,20,.3];b=basis(pose[3],40);points=np.asarray(pose[:3])+np.stack([b[2]*10,b[2]*20+b[0],b[2]*10+b[0]])
    uv,z,good=project(points,pose,40,width=100)
    np.testing.assert_allclose(uv[:,0],[50,52.5,55],atol=1e-10)
    assert good.all()

def test_multiview_occlusion_near_point_wins():
    frame=render(np.array([[0,0,0],[0,0,5.]]),np.array([[255,0,0],[0,255,0]],np.uint8),[0,0,10,0],0,width=32)
    assert frame['winner'][16,16]==1
    assert frame['depth'][16,16]==5
    assert frame['rgb'][16,16].tolist()==[0,255,0]

def test_multiview_invalid_stays_unknown_and_splat_preserves_rgb():
    xyz=np.array([[0,0,0],[0,0,11.]])
    rgb=np.array([[18,90,201],[255,255,255]],np.uint8)
    frame=render(xyz,rgb,[0,0,10,0],0,width=32,radius=1)
    assert frame['valid'].sum()==9
    assert (frame['winner'][frame['valid']]==0).all()
    assert np.isnan(frame['depth'][~frame['valid']]).all()
    assert (frame['rgb'][frame['valid']]==rgb[0]).all()
    assert frame['reprojection_max']<=2**.5*1.5

def test_multiview_polygon_box_overlap_is_unknown():
    item={'contour':[[-1,-1],[1,-1],[1,1],[-1,1]],'position':[0,0,0],'dimension':[2,2,2]}
    xyz=np.array([[0,0,0],[0,0,4],[3,3,0]])
    assigned,names,counts,ambiguous=associate(xyz,{'0':item,'1':item})
    assert (assigned==-1).all() and ambiguous==1
    assert counts==[1,1]

def test_multiview_far_and_behind_rejected():
    _,_,good=project([[0,0,11],[0,0,-500],[0,0,0]],[0,0,10,0],0,width=32)
    assert good.tolist()==[False,False,True]

def test_multiview_spatial_index_matches_direct_polygon():
    from matplotlib.path import Path
    rng=np.random.default_rng(4);xyz=rng.uniform(-30,30,(1000,3));poly=[[-10,-9],[20,-9],[20,14],[-10,14]]
    item={'contour':poly,'position':[5,2.5,0],'dimension':[30,23,6]}
    assigned,_,counts,_=associate(xyz,{'0':item})
    expected=Path(poly).contains_points(xyz[:,:2],radius=1e-7)&(np.abs(xyz[:,2])<=3.5)
    np.testing.assert_array_equal(assigned==0,expected)
    assert counts==[int(expected.sum())]

def test_multiview_scatter_equals_exhaustive_splat_sort():
    rng=np.random.default_rng(1);xyz=rng.uniform([-20,-20,-2],[20,20,12],(3000,3));rgb=rng.integers(256,size=(3000,3),dtype=np.uint8);pose=[0,0,30,.7]
    for tilt in [0,20,40,60]:
        for radius in [0,1]:
            frame=render(xyz,rgb,pose,tilt,width=32,radius=radius);uv,z,good=project(xyz,pose,tilt,width=32);ids=np.flatnonzero(good);xy=np.floor(uv[ids]).astype(int);entries=[]
            for i,(x,y) in zip(ids,xy):
                for dy in range(-radius,radius+1):
                    for dx in range(-radius,radius+1):
                        if 0<=x+dx<32 and 0<=y+dy<32:entries.append(((y+dy)*32+x+dx,z[i],i))
            expected=np.full(32*32,-1,dtype=int)
            for pixel,depth,index in sorted(entries,key=lambda e:(e[0],e[1],e[2])):
                if expected[pixel]<0:expected[pixel]=index
            np.testing.assert_array_equal(frame['winner'].ravel(),expected)

def test_multiview_identity_ties_do_not_win_by_index():
    from experiments.multiview_rgb_grounding.match_pilot import identity
    assert identity([1,1,1],['0','1','2'],'0')==(1/3,2)
    assert identity([1,1,1],['0','1','2'],'2')==(1/3,2)
    assert identity([3,2,1],['0','1','2'],'1')==(0,2)

def test_multiview_unobserved_identity_is_unknown():
    from experiments.multiview_rgb_grounding.match_pilot import identity
    assert identity([3,2],['0','1'],'2')==(None,None)

def test_multiview_grid_uses_original_world_bounds():
    from experiments.multiview_rgb_grounding.geometry import world_grid
    public={'bounds':[100,200,300,600],'grid_xy':[[0,0],[.5,.5],[1,1]]}
    np.testing.assert_array_equal(world_grid(public),[[100,200],[200,400],[300,600]])
    assert np.linalg.norm(world_grid(public)[1]-np.array([200,400]))==0
