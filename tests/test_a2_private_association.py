from types import SimpleNamespace
import numpy as np
from experiments.a2_waldo_landmark_belief.data_protocol import split_target_groups
from experiments.a2_waldo_landmark_belief.private_labels import PrivateAssociationIndex


def test_only_same_category_same_partition_boxes_are_rank_negatives():
    ids=[i for i in range(100) if split_target_groups([dict(map_name='m',object_id=i)])[0]=='train'][:3]
    def obj(oid,kind,x):
        return dict(id=oid,object_type=kind,contour=[[x,0],[x+2,0],[x+2,2],[x,2]])
    objects={ids[0]:obj(ids[0],'Building',0),ids[1]:obj(ids[1],'Car',10),ids[2]:obj(ids[2],'Building',20)}
    index=PrivateAssociationIndex(SimpleNamespace(objects={'m':objects}))
    boxes=[dict(poly=np.array(o['contour']),kind=0 if o['object_type']=='Car' else 1) for o in objects.values()]
    truth,known,_=index.associate('m',boxes,ids[0],'train')
    assert truth.tolist()==[1.,0,0]
    assert known.tolist()==[True,False,True]


def test_heldout_instance_and_unassociated_detection_are_unknown():
    train=next(i for i in range(100) if split_target_groups([dict(map_name='m',object_id=i)])[0]=='train')
    dev=next(i for i in range(100) if split_target_groups([dict(map_name='m',object_id=i)])[0]=='dev')
    objects={train:dict(id=train,object_type='Car',contour=[[0,0],[2,0],[2,2],[0,2]]),
             dev:dict(id=dev,object_type='Car',contour=[[10,0],[12,0],[12,2],[10,2]])}
    index=PrivateAssociationIndex(SimpleNamespace(objects={'m':objects}))
    boxes=[dict(poly=np.array(objects[dev]['contour']),kind=0),dict(poly=np.array([[50,0],[52,0],[52,2],[50,2]]),kind=0)]
    truth,known,_=index.associate('m',boxes,train,'train')
    assert not truth.any() and not known.any()


def test_coarse_car_click_regions_use_unique_symmetric_anchor_not_segmentation_iou():
    ids=[i for i in range(100) if split_target_groups([dict(map_name='m',object_id=i)])[0]=='train'][:2]
    def circle(x):
        a=np.linspace(0,2*np.pi,16,endpoint=False)
        return np.stack((x+4*np.cos(a),4*np.sin(a)),-1).tolist()
    objects={ids[0]:dict(id=ids[0],object_type='Car',contour=circle(0),position=[0,0,0]),
             ids[1]:dict(id=ids[1],object_type='Car',contour=circle(12),position=[12,0,0])}
    index=PrivateAssociationIndex(SimpleNamespace(objects={'m':objects}))
    def box(x):return dict(poly=np.array([[x-2,-1],[x+2,-1],[x+2,1],[x-2,1]]),kind=0,confidence=.8)
    truth,known,details=index.associate('m',[box(0),box(12)],ids[0],'train')
    assert truth.tolist()==[1.,0] and known.all()
    assert details[0]['method']=='unique_coarse_car_anchor'
    # Two detections close to the same annotation remain ambiguous.
    _,known,_=index.associate('m',[box(-.2),box(.2)],ids[0],'train')
    assert not known.any()


def test_coarse_car_anchor_requires_position_consistency_and_single_instance_box():
    ids=[i for i in range(100) if split_target_groups([dict(map_name='m',object_id=i)])[0]=='train'][:2]
    def obj(oid,x,position):
        a=np.linspace(0,2*np.pi,16,endpoint=False)
        return dict(id=oid,object_type='Car',contour=np.stack((x+4*np.cos(a),4*np.sin(a)),-1).tolist(),position=position)
    box=dict(poly=np.array([[-2.5,-1.5],[2.5,-1.5],[2.5,1.5],[-2.5,1.5]]),kind=0,confidence=.8)
    index=PrivateAssociationIndex(SimpleNamespace(objects={'m':{ids[0]:obj(ids[0],0,[30,0,0])}}))
    _,known,_=index.associate('m',[box],ids[0],'train')
    assert not known.any()
    index=PrivateAssociationIndex(SimpleNamespace(objects={'m':{ids[0]:obj(ids[0],0,[0,0,0]),ids[1]:obj(ids[1],2,[2,0,0])}}))
    _,known,_=index.associate('m',[box],ids[0],'train')
    assert not known.any()
