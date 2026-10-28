import copy
import numpy as np
import pytest
from experiments.a2_waldo_landmark_belief.relation_v3_verify import compare_packets,compare_metrics,validate_epoch_state,protected_paths,validate_verified_proof


def packet():
    return dict(scores=np.asarray([[.8,.1,.1]]),available=np.asarray([True]),is_real=np.asarray([True]),
        before_ne=np.asarray([20.000001]),after_ne=np.asarray([19.999999]),proposal_ne=np.asarray([10.]))


def test_verifier_rejects_success_boundary_flip_even_inside_distance_tolerance():
    old=packet();new=copy.deepcopy(old);compare_packets(old,new)
    new['before_ne'][0]=19.999999
    with pytest.raises(ValueError):compare_packets(old,new)
    new=copy.deepcopy(old);new['scores'][0]=[.1,.8,.1]
    with pytest.raises(ValueError):compare_packets(old,new)


def test_verifier_ignores_elapsed_time_but_rejects_changed_counts_or_missing_metrics():
    old=dict(seconds=1.,corrected=4,real=dict(views=20,proposal_ne=30.))
    new=dict(seconds=10.,corrected=4,real=dict(views=20,proposal_ne=30.))
    compare_metrics(old,new)
    new['corrected']=5
    with pytest.raises(ValueError):compare_metrics(old,new)
    with pytest.raises(ValueError):compare_metrics(old,dict(seconds=1.))


def test_epoch_proof_rejects_recipe_epoch_and_eligibility_mutations(tmp_path):
    import torch
    from experiments.a2_waldo_landmark_belief.relation_v3_train import fingerprint
    weights={'base.a':torch.ones(2)};record=dict(epoch=1,weights_sha256=fingerprint(weights),dev=dict(eligible=False))
    signature=dict(config=dict(seed=0),a0_sha256='original')
    state=dict(signature,epoch=1,model=weights,model_config=dict(text_dim=8),history=[record],checkpoint_eligible=False,
        stage_initialization=None,trainable_parameter_names=['base.a'],optimizer=dict(param_groups=[dict(params=[0])],state={0:dict(step=torch.tensor(4180.))}))
    args=([record],signature,tmp_path,['base.a'],dict(text_dim=8))
    validate_epoch_state(state,record,*args)
    for key,value in (('config',dict(seed=1)),('epoch',2),('checkpoint_eligible',True)):
        bad=copy.deepcopy(state);bad[key]=value
        with pytest.raises(ValueError):validate_epoch_state(bad,record,*args)


def test_retained_proof_detects_corrupted_nonselected_epoch_and_TensorBoard(tmp_path):
    import json,shutil
    from experiments.a2_waldo_landmark_belief.paths import sha
    for name in ('protocol.json','history.json','training_complete.json','dev_observations.json'):(tmp_path/name).write_text('{}')
    for epoch in range(1,26):
        (tmp_path/f'epoch_{epoch:02d}.pt').write_text(f'epoch{epoch}')
        (tmp_path/f'dev_epoch_{epoch:02d}.npz').write_text(f'packet{epoch}')
    for source,dest in (('epoch_25.pt','best.pt'),('epoch_25.pt','last.pt'),('epoch_01.pt','best_rank.pt')):shutil.copyfile(tmp_path/source,tmp_path/dest)
    (tmp_path/'tensorboard').mkdir();event=tmp_path/'tensorboard/events.out.tfevents.test';event.write_text('original events')
    proof=dict(status='v3_all25_artifacts_and_dev_verified',code_sha256={'x':'y'},roles_sha256='r',all_optimizer_steps_verified=True,
        all_dev_epochs_recomputed=True,selected_epoch=25,rank_selected_epoch=1,
        artifact_sha256={n:sha(tmp_path/n) for n in protected_paths(tmp_path)},
        epochs=[dict(epoch=e,checkpoint_sha256=sha(tmp_path/f'epoch_{e:02d}.pt'),packet_sha256=sha(tmp_path/f'dev_epoch_{e:02d}.npz')) for e in range(1,26)])
    (tmp_path/'artifact_verification.json').write_text(json.dumps(proof));validate_verified_proof(tmp_path,{'x':'y'},'r')
    path=tmp_path/'epoch_03.pt';old=path.read_text();path.write_text('corrupt')
    with pytest.raises(ValueError):validate_verified_proof(tmp_path,{'x':'y'},'r')
    path.write_text(old);event.write_text('modified events')
    with pytest.raises(ValueError):validate_verified_proof(tmp_path,{'x':'y'},'r')
