import numpy as np
from graphnav.training_retarget_proposals import registered_core_proposals


def test_training_proposals_keep_only_supported_unique_instance_cores():
    world = np.array([[0, 0], [1, 0], [5, 0], [20, 0]], float)
    result = registered_core_proposals(world, np.array([True, False, True, True]),
                                       np.array([[0, 0], [5, 0]], float), ['a', 'b'])
    assert result == {'a': [0], 'b': [2]}


def test_tied_instances_do_not_get_forced_id():
    result = registered_core_proposals(np.array([[0, 0]], float), np.ones(1, bool),
        np.array([[-.1, 0], [.1, 0]], float), ['a', 'b'])
    assert result == {}


def test_empty_registry_abstains():
    assert registered_core_proposals(np.array([[0, 0]], float), np.ones(1, bool), np.empty((0, 2)), []) == {}
