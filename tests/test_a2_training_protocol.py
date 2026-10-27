import json
import pytest
import torch
from experiments.a2_waldo_landmark_belief.training_protocol import validate_complete_cache, ranking_metrics, selection_key


def test_smoke_or_missing_episode_cache_cannot_authorize_full_training(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(complete=False)))
    with pytest.raises(ValueError,match='complete'):validate_complete_cache(tmp_path,dict(entries=[]))
    (tmp_path/'signature.json').write_text('{}')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(complete=True,entries=[])))
    source=dict(entries=[dict(index=0,split='train')])
    with pytest.raises(ValueError,match='episode'):validate_complete_cache(tmp_path,source)


def test_dev_ranking_metric_requires_real_positive_and_known_negative():
    scores=torch.tensor([[1.,2.,100.],[1.,2.,3.]])
    truth=torch.tensor([[1.,0,0],[1.,0,0]])
    known=torch.tensor([[True,True,False],[True,False,False]])
    reciprocal,correct=ranking_metrics(scores,truth,known,torch.ones_like(known))
    assert reciprocal==[.5] and correct==[0.]
    # An unassociated high-score box cannot be called a known negative.
    assert len(reciprocal)==1


def test_checkpoint_selection_does_not_claim_eligibility_without_dev_contrasts():
    eligible=dict(known_candidate_mrr=.6,offset_mae_m=1.,loss=2.,informative_views=12)
    empty=dict(known_candidate_mrr=None,offset_mae_m=None,loss=0.,informative_views=0)
    assert selection_key(eligible)>selection_key(empty)


def test_training_loop_visits_all_rows_updates_new_weights_and_records_dev_breaks():
    from torch.utils.data import DataLoader
    from experiments.a2_waldo_landmark_belief.model import LandmarkBeliefNetwork
    from experiments.a2_waldo_landmark_belief.batching import collate_views
    from experiments.a2_waldo_landmark_belief.train import run_epoch
    torch.manual_seed(4)
    def row():
        return dict(public=dict(network=dict(
            instruction_tokens=torch.randn(4,12),token_valid=torch.ones(4,dtype=torch.bool),
            mention_mask=torch.ones(2,4,dtype=torch.bool),landmark_name_tokens=torch.randn(2,3,12),
            name_valid=torch.ones(2,3,dtype=torch.bool),link_valid=torch.ones(2,dtype=torch.bool),
            box_rgb=torch.randn(3,16),box_valid=torch.ones(3,dtype=torch.bool),
            pair_geometry=torch.randn(3,2,14),pair_valid=torch.ones(3,2,dtype=torch.bool),offset_limit=torch.ones(3,2)*4),
            spatial=dict(base_logits=torch.tensor([0.,2.,-1.,-2.]),grid_world=torch.tensor([[0.,0.],[50.,0.],[100.,0.],[150.,0.]]),
                         candidate_xy=torch.tensor([[0.,0.],[50.,0.],[100.,0.]]),candidate_support=torch.eye(4,dtype=torch.bool)[:3])),
            labels=dict(ranking=torch.tensor([1.,0,0]),rank_label_mask=torch.tensor([True,True,False]),
                        angle=torch.tensor([[1.,0.],[0.,1.]]),angle_valid=torch.ones(2,dtype=torch.bool),distance=torch.ones(2),
                        offset=torch.zeros(3,2),offset_valid=torch.tensor([True,False,False]),
                        match=torch.tensor(1.),match_valid=torch.tensor(True),target_xy=torch.zeros(2)))
    rows=[row(),row()];loader=DataLoader(rows,batch_size=2,collate_fn=collate_views)
    model=LandmarkBeliefNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2)
    before=model.scorer.rank_head[-1].weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    trained=run_epoch(model,loader,optimizer=optimizer,refine=True,device='cpu')
    assert trained['views']==2 and trained['informative_views']==2 and trained['optimizer_updates']==1
    assert not torch.equal(before,model.scorer.rank_head[-1].weight)
    validation=run_epoch(model,loader,refine=True,device='cpu')
    assert validation['views']==2 and validation['internal_dev_probe']['a0_correct']==0


def test_last_epoch_resume_restores_history_and_all_tensorboard_steps(tmp_path):
    from experiments.a2_waldo_landmark_belief.train import restore_training_logs
    history=[dict(epoch=i,train=dict(loss=float(i)),dev=dict(loss=float(i)/2)) for i in range(1,26)]
    (tmp_path/'history.json').write_text(json.dumps(history[:-1]))
    class Writer:
        def __init__(self):self.rows=[];self.flushed=False
        def add_scalar(self,tag,value,step):self.rows.append((tag,value,step))
        def flush(self):self.flushed=True
    writer=Writer();restore_training_logs(history,tmp_path,writer)
    assert len(json.loads((tmp_path/'history.json').read_text()))==25
    assert [step for tag,value,step in writer.rows if tag=='Dev/loss']==list(range(1,26))
    assert writer.flushed


def test_resume_discards_best_alias_written_ahead_of_last_history(tmp_path,monkeypatch):
    from experiments.a2_waldo_landmark_belief import train
    monkeypatch.setattr(train,'runtime_hashes',lambda:{})
    monkeypatch.setattr(train,'training_hashes',lambda:{})
    history=[]
    for epoch in range(1,26):
        mrr=.9 if epoch==17 else .8 if epoch==24 else .5
        history.append(dict(epoch=epoch,dev=dict(known_candidate_mrr=mrr,offset_mae_m=1.,loss=1.)))
        torch.save(dict(epoch=epoch,runtime_code_sha256={},training_modules_sha256={}),tmp_path/f'epoch_{epoch:02d}.pt')
    torch.save(dict(epoch=99),tmp_path/'best.pt')
    train.reconcile_best_aliases(history,tmp_path)
    assert torch.load(tmp_path/'best_rank.pt',weights_only=False)['epoch']==17
    assert torch.load(tmp_path/'best.pt',weights_only=False)['epoch']==24
