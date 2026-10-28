"""Old sealed observations must get real rebuilt public inputs and new inference."""
import copy
import importlib
import pytest
import torch
from experiments.a2_waldo_landmark_belief.relation_v3_model import RelationV3Network
from tests.test_a2_reference_rebuild import fixture as pack_fixture,encode_names
from tests_confirmation_v4.test_replay_context import fixture as context_fixture


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_sealed_replay')


def fixture(tmp_path,monkeypatch):
    context_module,query,_=context_fixture(tmp_path,monkeypatch)
    from tests_confirmation_v4.test_replay_context import Tokenizer
    context=context_module.ConfirmationReferenceContext(Tokenizer(),[query])
    pack,meta,_=pack_fixture()
    pack['spatial'].update(base_logits=torch.tensor([.4,.3,.2,.1]).log(),
        grid_world=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]),candidate_xy=torch.tensor([[.5,.5]]),
        candidate_support=torch.ones(1,4,dtype=torch.bool))
    row=dict(public=pack,meta=meta,labels=dict(target_xy=torch.tensor([0.,0.])),
        target_instance=[query['map_name'],query['object_id']],fold='verify',episode=query['index'],step=0,
        prediction={'old_prediction_must_not_be_used':torch.tensor(float('nan'))},
        risk={'old_risk_must_not_be_used':torch.tensor(float('nan'))})
    torch.manual_seed(0);model=RelationV3Network(text_dim=768,vision_dim=8,hidden=8,heads=2).eval().requires_grad_(False)
    return row,query,context,model


def infer(row,query,context,model):
    return module().infer_row(row,model,'new signature','original bytes',query=query,context=context,
                              encode_names=encode_names,device='cpu')


def test_actual_rebuilt_road_inputs_and_new_predictions_replace_old_predictions(tmp_path,monkeypatch):
    row,query,context,model=fixture(tmp_path,monkeypatch);before=copy.deepcopy(row['public'])
    result=infer(row,query,context,model)
    assert result['meta']['valid_references']==1 and 'region' in result['meta']['references'][0]
    assert result['public']['network']['link_valid'].tolist()==[True]
    assert result['public']['network']['mention_mask'].tolist()==[[True,True]]
    assert 'scores' in result['prediction'] and torch.isfinite(result['risk']['scores']).all()
    assert result['original_raw_sha256']=='original bytes'
    for group in before:
        for key in before[group]:assert torch.equal(before[group][key],row['public'][group][key])
    assert row['meta']['references']==[]


def test_private_target_cannot_change_rebuilt_public_inputs_or_new_predictions(tmp_path,monkeypatch):
    row,query,context,model=fixture(tmp_path,monkeypatch);first=infer(row,query,context,model)
    changed=copy.deepcopy(row);changed['labels']['target_xy']=torch.tensor([999.,-999.])
    second=infer(changed,query,context,model)
    assert not torch.equal(first['labels']['target_xy'],second['labels']['target_xy'])
    for group in ('prediction','risk'):
        for key in first[group]:assert torch.equal(first[group][key],second[group][key])
    for group in first['public']:
        for key in first['public'][group]:assert torch.equal(first['public'][group][key],second['public'][group][key])


def test_sealed_read_requires_frozen_choice_before_tensor_deserialization(tmp_path,monkeypatch):
    monkeypatch.setattr(torch,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Premature sealed read')))
    with pytest.raises(ValueError):module().read_sealed_original(tmp_path,{}, {},'sig',authorized=False)


def test_wrong_query_is_rejected_before_model_or_encoder(tmp_path,monkeypatch):
    row,query,context,model=fixture(tmp_path,monkeypatch)
    def forbidden(*args,**kwargs):raise AssertionError('Wrong query must not reach forward')
    monkeypatch.setattr(model,'forward',forbidden)
    with pytest.raises(ValueError):module().infer_row(row,model,'sig','raw',query=dict(query,index=99),
        context=context,encode_names=forbidden,device='cpu')


def test_destination_tensor_is_first_accessed_after_public_forward(tmp_path,monkeypatch):
    row,query,context,model=fixture(tmp_path,monkeypatch);m=module();finished=[]
    runtime=m.runtime_forward
    def forward(*args,**kwargs):
        result=runtime(*args,**kwargs);finished.append(True);return result
    class DelayedLabels(dict):
        def __getitem__(self,key):
            assert finished, 'Private target accessed before public forward completed'
            return super().__getitem__(key)
    row['labels']=DelayedLabels(row['labels']);monkeypatch.setattr(m,'runtime_forward',forward)
    result=infer(row,query,context,model)
    assert finished and torch.equal(result['labels']['target_xy'],torch.tensor([0.,0.]))
