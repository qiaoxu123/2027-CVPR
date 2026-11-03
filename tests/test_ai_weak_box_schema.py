import pytest
from scripts.ai_weak_box_schema import validate_weak_box_record


def record():
    return dict(image_sha256='abc',human_verified=False,pixel_gold=False,annotation_only=True,
        assessment='candidate',target_unit='single_vehicle_proposal',boxes_px=[[10,20,30,40]],
        eligible_for_instance_loss=False,eligible_for_goal_loss=False)


def test_native_coordinates_sha_and_explicit_weak_status():
    r=record();assert validate_weak_box_record(r,'abc')==[[10/384,20/384,30/384,40/384]]
    with pytest.raises(ValueError):validate_weak_box_record(r,'changed')
    r['pixel_gold']=True
    with pytest.raises(ValueError):validate_weak_box_record(r,'abc')


def test_group_unknown_and_invalid_boxes_cannot_leak_into_instance_training():
    r=record();r['target_unit']='vehicle_group'
    with pytest.raises(ValueError):validate_weak_box_record(r,'abc')
    r=record();r['assessment']='ambiguous'
    with pytest.raises(ValueError):validate_weak_box_record(r,'abc')
    r=record();r['boxes_px']=[[10,20,500,40]]
    with pytest.raises(ValueError):validate_weak_box_record(r,'abc')
    r=record();r['eligible_for_instance_loss']=True
    with pytest.raises(ValueError):validate_weak_box_record(r,'abc')


def test_group_region_is_not_normalized_as_single_instance_and_hd_requires_native_metadata():
    r=record();r.update(target_unit='vehicle_group',assessment='group_target',boxes_px=[],target_count=2,
        group_region_bbox_px=[10,20,60,70],group_region_not_instance_bbox=True)
    assert validate_weak_box_record(r,'abc')==[]
    r['group_region_bbox_px']=[10,20,500,70]
    with pytest.raises(ValueError):validate_weak_box_record(r,'abc')
    with pytest.raises(ValueError):validate_weak_box_record(record(),'abc',image_shape=(800,800))
