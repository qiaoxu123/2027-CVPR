from experiments.a2_waldo_landmark_belief.revision_train import revision_selection


def test_revision_selection_prefers_full_proposal_and_net_map_gain_over_known_mrr():
    a=dict(proposal_r1=.2,proposal_ne=50.,map_r1=.7,broken=2,map_ne=30.,known_candidate_mrr=.99)
    b=dict(proposal_r1=.3,proposal_ne=40.,map_r1=.71,broken=1,map_ne=29.,known_candidate_mrr=.4)
    assert revision_selection(b,refine=False)>revision_selection(a,refine=False)
    assert revision_selection(b,refine=True)>revision_selection(a,refine=True)


def test_revision_epoch_processes_all_rows_and_reports_actual_map_net_changes():
    import torch
    from torch.utils.data import DataLoader
    from tests.test_a2_revision_loss import batch
    from experiments.a2_waldo_landmark_belief.batching import collate_views
    from experiments.a2_waldo_landmark_belief.revision_model import RevisionNetwork
    from experiments.a2_waldo_landmark_belief.revision_train import run_epoch
    b=batch();rows=[dict(public=dict(network={k:v[i] for k,v in b['network'].items()},spatial={k:v[i] for k,v in b['spatial'].items()}),labels={k:v[i] for k,v in b['labels'].items()}) for i in range(2)]
    data=DataLoader(rows,batch_size=1,collate_fn=collate_views)
    model=RevisionNetwork(text_dim=12,vision_dim=16,hidden=16,heads=2)
    optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=.001)
    metrics=run_epoch(model,data,optimizer=optimizer,refine=True,device='cpu')
    assert metrics['views']==2 and metrics['optimizer_updates']==2
    assert metrics['map_correct']==metrics['a0_correct']+metrics['corrected']-metrics['broken']
    assert 0<=metrics['proposal_r1']<=1 and 0<=metrics['map_r1']<=1


def test_revision_best_alias_recovery_discards_ahead_of_history_checkpoint(tmp_path):
    import torch
    from experiments.a2_waldo_landmark_belief.revision_train import reconcile
    history=[];hashes={'fake':'abc'}
    for epoch in range(1,26):
      r=dict(proposal_r1=.8 if epoch==7 else .5,proposal_ne=50.,map_r1=.9 if epoch==23 else .7,broken=1,map_ne=30.)
      history.append(dict(epoch=epoch,dev=r))
      torch.save(dict(epoch=epoch,code_sha256=hashes),tmp_path/f'epoch_{epoch:02d}.pt')
    torch.save(dict(epoch=99),tmp_path/'best.pt')
    reconcile(history,tmp_path,hashes)
    assert torch.load(tmp_path/'best.pt',weights_only=False)['epoch']==23
    assert torch.load(tmp_path/'best_rank.pt',weights_only=False)['epoch']==7
