import pytest
import torch
from experiments.bidirectional_core.model import CoreModel, collate_core, core_losses


def row(n=3):
    return {'public':dict(static_logits=torch.arange(8).float(),
        static_context=torch.randn(8,16), grid_xy=torch.rand(8,2),
        bounds=torch.tensor([0.,0.,200.,200.]), pose_world=torch.tensor([20.,20.,50.,0.]),
        ground_level=torch.tensor(0.),native_text=torch.randn(1024),
        roi_features=torch.randn(n,1024),roi_grid_ids=torch.arange(n),
        roi_valid=torch.ones(n,dtype=torch.bool),roi_quality=torch.ones(n,3)),
        'labels':dict(target_world=torch.tensor([100.,100.,50.]),
        teacher_world=torch.tensor([[100.,100.,50.]]).repeat(8,1),valid_teacher_z=torch.ones(8),
        positive_roi_mask=torch.tensor([True]+[False]*(n-1)) if n else torch.zeros(0,dtype=torch.bool),
        semantic_valid_mask=torch.ones(n,dtype=torch.bool))}


def test_padding_and_unknown_static_preserved():
    public,labels,_=collate_core([row(),row(0)])
    model=CoreModel(hidden=16,top_k=4)
    model.visual_gain.data.fill_(.4)
    output=model(public)
    assert torch.equal(output.belief_logits[~output.observed],public['static_logits'][~output.observed])
    assert not output.observed[1].any()
    assert output.trajectory.paths_world.shape==(2,4,3,8,3)


def test_labels_cannot_enter_forward():
    public,_,_=collate_core([row()]);public['target_world']=torch.ones(1,3)
    with pytest.raises(ValueError,match='private labels forbidden'):
        CoreModel(hidden=16)(public)


def test_multipos_gradient_and_unknown_exclusion():
    public,labels,_=collate_core([row()]); model=CoreModel(hidden=16,top_k=4)
    out=model(public)
    labels['semantic_valid_mask'][0,2]=False
    loss=core_losses(out,public,labels,vl_weight=.1)
    loss['total'].backward()
    assert torch.isfinite(loss['total'])
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.matcher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.trajectory.parameters())
    scores=out.roi_logits.detach().clone().requires_grad_();out.roi_logits=scores
    visual=core_losses(out,public,labels,vl_weight=.1)['visual_language']
    visual.backward()
    assert scores.grad[0,2]==0


def test_all_positive_or_no_support_is_not_false_negative():
    public,labels,_=collate_core([row(),row(0)])
    labels['positive_roi_mask'][0]=True
    loss=core_losses(CoreModel(hidden=16,top_k=4)(public),public,labels,vl_weight=.1)
    assert loss['visual_language_eligible']==0
    assert loss['visual_language']==0


def test_invalid_grid_id_rejected():
    public,_,_=collate_core([row()]);public['roi_grid_ids'][0,0]=8
    with pytest.raises(ValueError,match='invalid observed grid'):
        CoreModel(hidden=16)(public)


def test_same_seed_head_initialization_is_identical():
    from experiments.bidirectional_core.train import tensor_state_sha
    torch.manual_seed(0);a=CoreModel(hidden=16)
    torch.manual_seed(0);b=CoreModel(hidden=16)
    assert tensor_state_sha(a.state_dict())==tensor_state_sha(b.state_dict())


def test_frozen_resource_mutation_rejected(tmp_path):
    import hashlib
    from experiments.bidirectional_core.train import verify_frozen_resources
    p=tmp_path/'checkpoint';p.write_bytes(b'original immutable checkpoint')
    manifest={'frozen_resources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()}}
    assert verify_frozen_resources(manifest)==manifest['frozen_resources']
    p.write_bytes(b'different weights')
    with pytest.raises(ValueError,match='frozen resource changed'):
        verify_frozen_resources(manifest)


def test_multicolor_building_query_and_support_remain_unknown():
    from experiments.bidirectional_core.cache import conservative_attributes,conservative_object_attributes
    assert conservative_attributes('gray and brown building','Building')=={}
    assert conservative_object_attributes([{'target':'gray and brown building'}],'Building')=={}
    assert conservative_attributes('a white roof','Building')=={'roof_color':'white'}


def test_reverse_without_launch_identity_and_invisible_target_have_no_vl():
    from experiments.bidirectional_core.cache import supervision
    from types import SimpleNamespace
    roi=[{'object_id':'neighbor','object_type':'Building'}]
    ep={'direction':'reverse','private_labels':{'target_object_id':None},'reverse_geometry_description':{'label_positive_object_ids':['neighbor']}}
    pos,judged=supervision(ep,None,roi,{}, {}, {})
    assert not pos.any() and not judged.any()
    ep['private_labels']['target_object_id']='unobserved_target'
    pos,judged=supervision(ep,None,roi,{}, {}, {})
    assert not pos.any() and not judged.any()


def test_cached_budget_forward_repeat_matches_bidirectional_steps(tmp_path):
    import json
    from graphnav.checkpointing import file_sha256
    from experiments.bidirectional_core.cache import CachedCoreDataset
    rows=[]
    full=row(1);static_keys={'static_logits','static_context','grid_xy','bounds','native_text'}
    queries=[{k:v for k,v in full['public'].items() if k in static_keys}]
    for role in ('forward','reverse'):
        rows.append(dict(query_index=0,public={k:v for k,v in full['public'].items() if k not in static_keys},
            labels=full['labels'],metadata={'sample_id':role,'direction':role}))
    torch.save(queries,tmp_path/'queries.pt');torch.save(rows,tmp_path/'rows.pt');(tmp_path/'queries.jsonl').write_text('{}\n')
    manifest={'complete':True,'chunks':['rows.pt'],'files':{p:file_sha256(tmp_path/p) for p in ('queries.pt','queries.jsonl','rows.pt')},
        'matched_budget':[{'E0_forward_sample_id':'forward','E1_sample_id':r} for r in ('forward','reverse')]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    e0=CachedCoreDataset(tmp_path,'E0');e1=CachedCoreDataset(tmp_path,'E1')
    assert len(e0)==len(e1)==2
    assert [e0[i]['metadata']['direction'] for i in range(2)]==['forward','forward']
    assert [e1[i]['metadata']['direction'] for i in range(2)]==['forward','reverse']
    assert set(e1[0]['public'])==CoreModel.PUBLIC_KEYS
    (tmp_path/'rows.pt').write_bytes(b'corrupt cache')
    with pytest.raises(ValueError,match='cache changed'):
        CachedCoreDataset(tmp_path)


def test_forward_unconfirmed_target_cannot_train_using_other_same_color():
    from experiments.bidirectional_core.cache import supervision
    from types import SimpleNamespace
    ep={'direction':'forward','map_name':'m','private_labels':{'target_object_id':'target'}}
    regions=[{'object_id':'target','object_type':'Building'},{'object_id':'other','object_type':'Building'}]
    objects={'m':{'target':{'object_type':'Building'}}};processed={'m':{'target':[{'target':'white roof'}]}}
    attrs={('m','other'):{'roof_color':{'label':'white'}}}
    pos,judged=supervision(ep,SimpleNamespace(episode_id=('m',0,0)),regions,attrs,processed,objects)
    assert not pos.any() and not judged.any()


def test_forward_multipos_and_unknown_supported_attributes():
    from experiments.bidirectional_core.cache import supervision
    from types import SimpleNamespace
    ep={'direction':'forward','map_name':'m','private_labels':{'target_object_id':'target'}}
    regions=[{'object_id':oid,'object_type':'Building'} for oid in ('target','other','negative','unknown')]
    objects={'m':{'target':{'object_type':'Building'}}};processed={'m':{'target':[{'target':'white roof'}]}}
    attrs={('m',oid):{'roof_color':{'label':color}} for oid,color in [('target','white'),('other','white'),('negative','gray')]}
    pos,judged=supervision(ep,SimpleNamespace(episode_id=('m',0,0)),regions,attrs,processed,objects)
    assert pos.tolist()==[True,True,False,False]
    assert judged.tolist()==[True,True,True,False]
