import copy
import pytest
import torch
from experiments.a2_waldo_landmark_belief.relation_v3_model import RelationV3Network
from experiments.a2_waldo_landmark_belief.relation_v3_train import start_refinement,verify_optimizer_steps,require_idle_gpu,validate_resume,fingerprint


def test_refinement_reloads_selected_rank_weights_and_resets_optimizer():
    torch.manual_seed(0);model=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2)
    saved=copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for p in model.parameters():p.add_(10)
    optimizer=start_refinement(model,dict(model=saved))
    assert not optimizer.state and all(torch.equal(v,saved[k]) for k,v in model.state_dict().items())
    assert all(p.requires_grad for group in optimizer.param_groups for p in group['params'])


def test_optimizer_proof_requires_all_and_only_stage_active_parameters():
    state=dict(epoch=2,trainable_parameter_names=['base.a','risk.b'],optimizer=dict(param_groups=[dict(params=[0,1])],state={0:dict(step=torch.tensor(8360.))}))
    verify_optimizer_steps(state)
    bad=copy.deepcopy(state);bad['optimizer']['state'][0]['step']-=1
    with pytest.raises(ValueError):verify_optimizer_steps(bad)
    bad=copy.deepcopy(state);bad['optimizer']['state'][1]=dict(step=torch.tensor(8360.))
    with pytest.raises(ValueError):verify_optimizer_steps(bad)
    state['epoch']=21;state['optimizer']['state']={0:dict(step=torch.tensor(4180.)),1:dict(step=torch.tensor(4180.))}
    verify_optimizer_steps(state)


def test_busy_gpu_fails_before_any_CUDA_work(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'1234\n')
    with pytest.raises(ValueError,match='occupied'):require_idle_gpu()


def test_resume_binds_actual_weights_and_last_bytes_to_saved_epoch(tmp_path):
    import shutil
    weights={'base.a':torch.ones(2)}
    state=dict(binding='a',epoch=2,model=weights,trainable_parameter_names=['base.a'],
        optimizer=dict(param_groups=[dict(params=[0])],state={0:dict(step=torch.tensor(8360.))}),
        history=[dict(epoch=1),dict(epoch=2,weights_sha256=fingerprint(weights))])
    epoch=tmp_path/'epoch_02.pt';torch.save(state,epoch);shutil.copyfile(epoch,tmp_path/'last.pt')
    validate_resume(state,tmp_path,dict(binding='a'))
    corrupt=copy.deepcopy(state);corrupt['model']['base.a']+=1
    torch.save(corrupt,tmp_path/'last.pt')
    with pytest.raises(ValueError):validate_resume(corrupt,tmp_path,dict(binding='a'))
