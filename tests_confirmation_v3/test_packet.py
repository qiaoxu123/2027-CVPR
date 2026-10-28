import copy
import pytest
import torch
from tests_confirmation_v3.test_capture import policy
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold
from experiments.a2_waldo_landmark_belief.relation_v3_confirmation import capture_public_row
from experiments.a2_waldo_landmark_belief.relation_v3_packet import make_packet


def row_and_model():
    p=policy();row=capture_public_row(p,episode=4,step=0,signature='sig',command=dict(xyz=[0,0,10],should_stop=False))
    target=next(i for i in range(100) if calibration_fold('m',i)=='fit')
    row.update(labels=dict(target_xy=torch.zeros(2)),target_instance=['m',target],fold='fit')
    return row,p.branch


def test_packet_preserves_actual_scores_and_fixed_action_for_private_scoring():
    row,model=row_and_model();p=make_packet(row,model)
    assert torch.equal(p['scores'],row['risk']['scores'])
    assert p['group']==row['target_instance'] and p['before_ne']==0
    assert p['action_ne']==float((row['public']['spatial']['grid_world'][row['risk']['fixed_action_logits'].argmax()]).norm())


def test_packet_rejects_inconsistent_proposal_logits_or_class_scores():
    row,model=row_and_model()
    for key in ('q','head_logits','scores','fixed_action_logits'):
        bad=copy.deepcopy(row);bad['risk'][key].fill_(.123)
        with pytest.raises(ValueError):make_packet(bad,model)


def test_near_tie_uses_saved_action_and_scores_even_when_cpu_argmax_differs():
    from experiments.a2_waldo_landmark_belief.relation_v3_runtime import evidence_forward
    row,model=row_and_model();q=row['risk']['q'];delta=(q[2]-q[0])/3
    row['public']['spatial']['base_logits']=torch.stack((.4+delta/2,q.new_tensor(.2),.4-delta/2)).log()
    packet=evidence_forward({k:v[None] for k,v in row['public']['spatial'].items()},
        {k:v[None] for k,v in row['prediction'].items()},model)
    row['risk']={k:v[0].clone() for k,v in packet.items()}
    fixed=row['risk']['fixed_action_logits'];cpu_winner=int(fixed.argmax());other=2 if cpu_winner==0 else 0
    assert cpu_winner in (0,2) and abs(float(fixed[0]-fixed[2]))<1e-6
    fixed[other]=fixed.max()+1e-7
    row['risk']['scores'][1]+=1e-7;row['risk']['scores'][0]-=1e-7
    assert int(fixed.argmax())==other
    actual=make_packet(row,model)
    assert torch.equal(actual['scores'],row['risk']['scores'])
    grid=row['public']['spatial']['grid_world'];target=row['labels']['target_xy']
    assert actual['action_ne']==float((grid[other]-target).norm())
    assert actual['action_ne']!=float((grid[cpu_winner]-target).norm())
