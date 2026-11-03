import torch
from scripts.train_world_relation_factor import forward_inputs, goal_loss, REFERENCE_INPUTS
from scripts.train_local_chain_position import INPUTS


def test_forward_whitelist_rejects_private_loss_fields():
    batch = {key: torch.ones(1) for key in (*INPUTS, *REFERENCE_INPUTS)}
    batch.update(candidate_world=torch.ones(1, 2, 2), target_world=torch.zeros(1, 2),
                 instance_labels=torch.ones(1), target_id='private')
    public = forward_inputs(batch, 'cpu')
    assert 'world_hypotheses' in public
    assert not {'target_world', 'instance_labels', 'target_id'} & public.keys()


def test_world_loss_rewards_mass_in_correct_region_and_is_finite():
    world = torch.tensor([[[0., 0.], [50., 0.]]])
    truth = torch.tensor([[0., 0.]])
    base = torch.zeros(1, 2)
    correct = torch.tensor([[2., 0.]], requires_grad=True)
    wrong = torch.tensor([[0., 2.]])
    good_loss = goal_loss(correct, base, world, truth)
    assert good_loss < goal_loss(wrong, base, world, truth)
    good_loss.backward()
    assert correct.grad[0, 0] < 0
