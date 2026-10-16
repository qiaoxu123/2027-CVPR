import json
from types import SimpleNamespace
import numpy as np
import torch
from graphnav.landmark_context import append_context, public_context
from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.full_localization import FullTokenGeometryLocalizer


class Encoder:
    def encode(self, texts):
        return torch.zeros(len(texts), 8)


def test_context_preserves_reference_fields_and_masked_forward(tmp_path):
    record = dict(map_name='map',instruction='near Hall',world_bounds=[0,0,100,100],
                  physical_map_sizes=[100,100],target_xy=[.5,.5],dataset_version='revised',
                  landmarks=[dict(name='Hall',object_type='Building',contour=[[.1,.1],[.2,.1],[.2,.2]])])
    p=tmp_path/'query.jsonl';p.write_text(json.dumps(record)+'\n')
    dataset=FullCandidateDataset(p,grid_size=28,max_landmarks=2,max_roads=0,max_relations=2)
    batch=collate_full([dataset[0]])
    keys=('nodes','node_types','node_valid','entity_valid','adjacency','edges','prior','prior_buildings','rules','relation_fields','pair_geometry','grid_xy')
    x={k:batch[k] for k in keys};names=torch.zeros(1,len(batch['node_names'][0]),8)
    extras=[dict(name='Other',object_type='Building',contour=[[.7,.7],[.8,.7],[.8,.8]])]
    expanded,en=append_context(x,names,record,extras,Encoder(),'all_named')
    masked,mn=append_context(x,names,record,extras,Encoder(),'all_named_masked')
    for key in ('prior','rules','relation_fields','prior_buildings','grid_xy'):
        assert expanded[key] is x[key]
    assert expanded['entity_valid'].sum()==x['entity_valid'].sum()+1
    assert masked['entity_valid'].sum()==x['entity_valid'].sum()
    torch.manual_seed(0)
    model=FullTokenGeometryLocalizer(hidden=8,text_dim=8,heads=2,max_relations=2,dropout=0).eval()
    tokens=torch.randn(1,3,8);valid=torch.ones(1,3,dtype=torch.bool)
    a=model(x,variant='geometry',tokens=tokens,token_mask=valid,node_emb=names)
    b=model(masked,variant='geometry',tokens=tokens,token_mask=valid,node_emb=mn)
    torch.testing.assert_close(a.belief_logits,b.belief_logits,rtol=0,atol=1e-6)
    torch.testing.assert_close(a.dense_context,b.dense_context,rtol=0,atol=1e-6)
    c=model(expanded,variant='geometry',tokens=tokens,token_mask=valid,node_emb=en)
    assert not torch.equal(a.belief_logits,c.belief_logits)


def test_public_catalog_only_and_no_reference_duplication():
    record=dict(map_name='map',world_bounds=[0,0,100,100],
                landmarks=[dict(name='Hall',contour=[[.1,.1],[.2,.1],[.2,.2]])])
    landmarks=[SimpleNamespace(name='Hall',object_id=1,contour=np.array([[10,10],[20,10],[20,20]])),
               SimpleNamespace(name='Other',object_id=2,contour=np.array([[70,70],[80,70],[80,80]]))]
    catalog=SimpleNamespace(all_landmarks=lambda name:landmarks,
                            objects={'map':{1:{'object_type':'Building'},2:{'object_type':'Building'}}})
    extras=public_context(record,catalog)
    assert len(extras)==1 and extras[0]['name']=='Other'
    assert set(extras[0])=={'name','object_type','contour'}
