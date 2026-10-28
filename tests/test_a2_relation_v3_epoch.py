import torch
from torch.utils.data import DataLoader
from tests.test_a2_relation_v3_loss import batch
from experiments.a2_waldo_landmark_belief.batching import collate_views
from experiments.a2_waldo_landmark_belief.relation_v3_model import RelationV3Network
from experiments.a2_waldo_landmark_belief.relation_v3_epoch import run_epoch


class Views(torch.utils.data.Dataset):
    entries=[dict(map_name='map',object_id=i,source_kind='teacher_snapshot' if i==0 else 'revision_v1_A0_closedloop') for i in range(2)]
    def __len__(self):return 2
    def __getitem__(self,i):
        b=batch();labels={k:v[i] for k,v in b['labels'].items()};labels['source_is_real']=torch.tensor(i==1)
        labels['sample_weight']=torch.tensor(8. if i else 1.)
        return dict(public=dict(network={k:v[i] for k,v in b['network'].items()},spatial={k:v[i] for k,v in b['spatial'].items()}),labels=labels)


def test_epoch_visits_all_rows_reports_real_teacher_and_records_same_Q_actions():
    torch.manual_seed(0);m=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2)
    loader=DataLoader(Views(),batch_size=2,collate_fn=collate_views)
    metrics,packets=run_epoch(m,loader,refine=True,device='cpu')
    assert metrics['views']==2 and metrics['optimizer_updates']==0
    assert metrics['real']['views']==metrics['teacher']['views']==1
    assert packets['scores'].shape==(2,3) and len(packets['identities'])==2
    assert packets['is_real'].tolist()==[False,True]
    assert m.training is False and all(p.grad is None for p in m.parameters())
    before=m.risk[-1].weight.detach().clone();optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
    metrics,packets=run_epoch(m,loader,optimizer=optimizer,refine=True,device='cpu')
    assert metrics['views']==2 and metrics['optimizer_updates']==1 and packets is None
    assert not torch.equal(before,m.risk[-1].weight)


def test_partial_loader_is_rejected_instead_of_becoming_an_epoch():
    import pytest
    m=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2)
    loader=DataLoader(Views(),batch_size=3,drop_last=True,collate_fn=collate_views)
    with pytest.raises(ValueError,match='Incomplete'):run_epoch(m,loader,device='cpu')
