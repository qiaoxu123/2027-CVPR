import torch,pytest
from experiments.a2_waldo_landmark_belief.revision_calibrate import revision_packet,evaluate_gate,fit_benefit_temperature
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold


def raw(group):
    return dict(target_instance=group,fold=calibration_fold(*group),episode=1,step=0,labels=dict(target_xy=torch.tensor([30.,0.])),
      public=dict(spatial=dict(base_logits=torch.tensor([.4,.3,.2,.1]).log(),grid_world=torch.tensor([[0.,0.],[10.,0.],[20.,0.],[30.,0.]]),candidate_xy=torch.tensor([[30.,0.]]),candidate_support=torch.tensor([[False,False,False,True]]))),
      prediction=dict(scores=torch.tensor([1.]),candidate_match_logits=torch.tensor([4.]),reliable=torch.tensor([True]),benefit_logit=torch.tensor(4.)))


def test_fixed_gate_calibration_uses_actual_zero_offset_global_mix():
    p=revision_packet(raw(['m',1]));summary=evaluate_gate([p],dict(scale=1.,threshold=0.),1.,device='cpu')
    assert summary['mean_r1_difference']==1 and summary['mean_ne_difference_m']==-30
    off=evaluate_gate([p],dict(scale=0.,threshold=1.),1.,device='cpu')
    assert off['mean_r1_difference']==0 and off['broken_groups']==0


def test_temperature_fitting_cannot_consume_verification_targets():
    oid=next(i for i in range(100) if calibration_fold('m',i)=='verify')
    with pytest.raises(ValueError):fit_benefit_temperature([revision_packet(raw(['m',oid]))])


def test_cached_calibration_packet_integrity_is_not_self_reported(tmp_path):
    from experiments.a2_waldo_landmark_belief.revision_calibrate import save_packet,load_packet
    path=tmp_path/'p.pt';p=dict(signature_sha256='sig',source_row_sha256='raw',q=torch.tensor([.2,.8]))
    save_packet(p,path);assert torch.equal(load_packet(path,'sig','raw')['q'],p['q'])
    p['q']=torch.tensor([.8,.2]);torch.save(p,path)
    with pytest.raises(ValueError,match='checksum'):load_packet(path,'sig','raw')
