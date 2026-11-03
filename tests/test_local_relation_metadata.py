from graphnav.local_relation_metadata import relation_metadata


def proposal(relation='left of', reverse=False):
    return {'reference_phrases': ['the grey car'], 'parse': {
        'target_token': 2, 'scope_uncertain': False, 'paths': [],
        'edges': [{'a': 9 if reverse else 2, 'b': 2 if reverse else 9,
                   'relation': relation}]}}


def test_left_right_remain_distinct_without_world_direction_claim():
    left = relation_metadata(proposal())
    right = relation_metadata(proposal('right of'))
    assert left['active'] and right['active']
    assert left['kind'] == 'left' and right['kind'] == 'right'
    assert left['frame'] == 'unspecified'
    assert 'world_direction' not in left


def test_reversed_edge_inverts_direction():
    assert relation_metadata(proposal(reverse=True))['kind'] == 'right'
    assert relation_metadata(proposal('in', reverse=True))['kind'] == 'contains'


def test_uncertain_scope_and_multiple_references_abstain():
    value = proposal(); value['parse']['scope_uncertain'] = True
    assert not relation_metadata(value)['active']
    value = proposal(); value['reference_phrases'].append('the black car')
    assert not relation_metadata(value)['active']
    assert not relation_metadata(proposal('between'))['active']


def test_public_path_not_confused_with_visual_reference():
    value = proposal('beside')
    value['parse']['paths'] = [{'nodes': [2, 4], 'hops': 1}]
    value['parse']['edges'].append({'a': 2, 'b': 4, 'relation': 'on'})
    assert relation_metadata(value)['kind'] == 'near'
    value['parse']['edges'].append({'a': 2, 'b': 15, 'relation': 'behind'})
    assert not relation_metadata(value)['active']


def test_missing_and_unsupported_proposals_abstain():
    assert not relation_metadata({})['active']
    assert not relation_metadata(proposal('north left of'))['active']
