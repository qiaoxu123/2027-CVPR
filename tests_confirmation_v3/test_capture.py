from types import SimpleNamespace
import pytest
import torch
from tests.test_a2_relation_v3_loss import batch
from experiments.a2_waldo_landmark_belief.relation_v3_model import RelationV3Network
from experiments.a2_waldo_landmark_belief.relation_v3_runtime import evidence_forward
from experiments.a2_waldo_landmark_belief.relation_v3_confirmation import capture_public_row,assert_shadow_OFF,commands_match


def policy():
    torch.manual_seed(0);b=batch();model=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2).eval()
    network={k:v[:1] for k,v in b['network'].items()};spatial={k:v[:1] for k,v in b['spatial'].items()};out=model(**network)
    field=SimpleNamespace(belief_logits=spatial['base_logits'])
    return SimpleNamespace(local_diagnostics=dict(a2_alpha=0.,a2_updated=False),original_field=field,corrected_field=field,
        branch=model,
        pack=dict(network={k:v[0] for k,v in network.items()},spatial={k:v[0] for k,v in spatial.items()}),
        branch_output=out,risk_output=evidence_forward(spatial,out,model),meta=dict(candidates=[]))


def test_capture_contains_actual_inference_without_any_destination_label():
    p=policy();row=capture_public_row(p,episode=4,step=0,signature='sig',command=dict(xyz=[0,0,10],should_stop=False))
    assert not {'labels','target_instance','fold','private'}&set(row)
    assert torch.equal(row['risk']['scores'],p.risk_output['scores'][0])
    assert row['risk']['fixed_action_logits'].shape==(3,)
    assert row['public']['network']['instruction_tokens'].shape==p.pack['network']['instruction_tokens'].shape


def test_shadow_capture_rejects_nonzero_gate_or_changed_belief():
    p=policy();p.local_diagnostics['a2_alpha']=.25
    with pytest.raises(ValueError):assert_shadow_OFF(p)
    p=policy();p.corrected_field=SimpleNamespace(belief_logits=p.original_field.belief_logits+.01)
    with pytest.raises(ValueError):assert_shadow_OFF(p)


def test_partial_replay_compares_effective_commands():
    assert commands_match(dict(xyz=[0,0,10],should_stop=True),dict(xyz=[100,100,20],should_stop=True))
    assert not commands_match(dict(xyz=[0,0,10],should_stop=False),dict(xyz=[100,100,20],should_stop=False))
    assert not commands_match(dict(xyz=[0,0,10],should_stop=True),dict(xyz=[0,0,10],should_stop=False))
