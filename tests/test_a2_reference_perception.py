from types import SimpleNamespace
import copy

import torch

from experiments.a2_waldo_landmark_belief.reference_perception import (
    ReferencePerception, bind_public_road_references,
)
from experiments.a2_waldo_landmark_belief.reference_rebuild import rebuild_reference_pack


def test_public_binding_uses_no_destination_and_does_not_replace_legacy_references():
    class Episode:
        map_name = 'map'
        reference_names = ['Road']
        @property
        def object_id(self): raise AssertionError('Private destination forbidden')
    contour = [[0,0],[1,0],[1,1],[0,1]]
    catalog = SimpleNamespace(objects={'map':{str(i):dict(id=i,name='Road',object_type=kind,
        contour=[[x+i-1,y] for x,y in contour]) for i,kind in [(1,'TrafficRoad'),(2,'Footpath')]}})
    policy = SimpleNamespace(a2_references=['legacy'],controller=object())
    before = policy.controller
    bind_public_road_references(policy,Episode(),catalog)
    assert policy.a2_references == ['legacy'] and policy.controller is before
    assert len(policy.a2_road_references) == 1 and policy.a2_road_references[0]['linked']


def test_live_adapter_calls_legacy_once_and_matches_offline_reference_features():
    from test_a2_reference_rebuild import fixture, encode_names
    pack,meta,refs = fixture()
    class Legacy:
        calls = 0
        def extract(self,*args):
            self.calls += 1
            return pack,meta
    bert = SimpleNamespace(max_length=96,encode_tokens=encode_names,
                           tokenizer=lambda text,**kwargs: {'offset_mapping':[[0,4],[5,9]]})
    policy = SimpleNamespace(record={'instruction':'Main Road'},bert=bert,a2_road_references=refs)
    legacy = Legacy()
    actual,info = ReferencePerception(legacy).extract(policy,object(),object())
    expected,expected_meta = rebuild_reference_pack(pack,meta,instruction='Main Road',offsets=[[0,4],[5,9]],
        references=refs,encode_names=encode_names)
    assert legacy.calls == 1
    for section in actual:
        for key in actual[section]: assert torch.equal(actual[section][key],expected[section][key])
    assert info['reference_surfaces'] == expected_meta['reference_surfaces']
    assert meta['references'] == []
