import pytest
from scripts.prepare_dynamic_evidence import causal_history


def test_causal_visual_history_never_uses_other_episode_or_future_frame():
    rows=[dict(source_episode_index=1,observed_step=step,observation_cutoff_step=step) for step in (0,8,16)]
    rows.append(dict(source_episode_index=2,observed_step=0,observation_cutoff_step=0))
    assert causal_history(rows,[0,1,2,3],episode_index=1,cutoff=8)==[0,1]
    rows[1]['observation_cutoff_step']=0
    with pytest.raises(ValueError,match='future'):causal_history(rows,[0,1,2],episode_index=1,cutoff=8)


def test_dense_cache_preserves_goal_ranking_precision():
    from types import SimpleNamespace
    import torch
    from scripts.cache_a3_features import dense_cache_values
    logits=torch.tensor([[1.0001,1.0004]],dtype=torch.float32)
    assert torch.equal(logits.half()[:,0],logits.half()[:,1])
    output=dense_cache_values(SimpleNamespace(belief_logits=logits,dense_context=torch.ones(1,2,3)))
    assert output['belief_logits'].dtype==torch.float32 and torch.equal(output['belief_logits'],logits)
    assert output['belief_logits'].argmax(-1).item()==1
