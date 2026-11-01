import json
from types import SimpleNamespace
import pytest
import torch
from graphnav.train_collection import verified_train_indices


def test_collection_rejects_group_leakage_even_when_episode_index_differs(tmp_path):
    data=[SimpleNamespace(map_name='a',instruction='same'),SimpleNamespace(map_name='a',instruction='same'),SimpleNamespace(map_name='b',instruction='other')]
    checkpoint=tmp_path/'base.pt';torch.save(dict(config=dict(dev_episodes=[0])),checkpoint)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(partition='train',status='complete',episode_indices=[1,2])))
    with pytest.raises(ValueError,match='Dev group'):verified_train_indices(data,manifest,checkpoint)
    manifest.write_text(json.dumps(dict(partition='train',status='complete',episode_indices=[2])))
    assert verified_train_indices(data,manifest,checkpoint)==[2]
    manifest.write_text(json.dumps(dict(partition='dev',status='complete',episode_indices=[2])))
    with pytest.raises(ValueError,match='Train partition'):verified_train_indices(data,manifest,checkpoint)
