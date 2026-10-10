from types import SimpleNamespace
import pytest
from graphnav.full_experiment import new_model
from scripts.evaluate_p1_counterfactuals import checkpoint_model,swap_cardinal_relation


def test_counterfactual_loader_respects_legacy_and_relative_checkpoint_architecture():
    for hidden,heads,relative in ((128,4,False),(256,8,True)):
        args=dict(hidden=hidden,heads=heads,max_relations=8,vision_dim=768,top_k=20,rule_weight=1.,relative_position=relative)
        source=new_model(SimpleNamespace(**args),'cpu')
        loaded=checkpoint_model(dict(args=args,model=source.state_dict()),'cpu')
        assert loaded.hidden==hidden and loaded.relative_position is relative
        assert loaded.pos[0].weight.equal(source.pos[0].weight)
    with pytest.raises(ValueError,match='architecture'):
        checkpoint_model({},'cpu')


def test_cardinal_swap_changes_relation_and_preserves_direction_in_landmark_name():
    assert swap_cardinal_relation('Find a building to the east of west hall.')=='Find a building to the west of west hall.'
    with pytest.raises(ValueError):swap_cardinal_relation('east hall near a building')
