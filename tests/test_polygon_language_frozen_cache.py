"""Guard the frozen core against stale cached Geometry across TRAIN batches."""
from types import SimpleNamespace

import torch

import scripts.train_polygon_language_attention as train


class FakeFrozenModel:
    def __init__(self):
        self.resets=0
        self.calls=0

    def reset_episode(self):
        self.resets+=1

    def forward_full_relation(self,**kwargs):
        self.calls+=1
        value=float(self.resets)
        return dict(
            relation_field=SimpleNamespace(belief_logits=torch.full((2,4),value)),
            geometry_field=SimpleNamespace(dense_context=torch.full((2,4,128),value)),
        )


def test_every_frozen_train_batch_resets_per_episode_cache(monkeypatch):
    def fake_inputs(b,bert,device):
        return dict(
            static_inputs=dict(pair_geometry=torch.zeros(2,4,3,10),
                               entity_valid=torch.ones(2,3,dtype=torch.bool)),
            node_emb=torch.zeros(2,3,768),tokens=torch.zeros(2,5,768),
            token_mask=torch.ones(2,5,dtype=torch.bool),
        )
    monkeypatch.setattr(train,"full_relation_inputs",fake_inputs)
    model=FakeFrozenModel()
    first=train.frozen_features(model,{},None,"cpu")
    second=train.frozen_features(model,{},None,"cpu")
    assert model.resets==2
    assert model.calls==2
    assert first["base_logits"][0,0]==1
    assert second["base_logits"][0,0]==2
