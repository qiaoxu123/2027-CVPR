"""Conservative public syntax proposals, not verified reference-instance labels.

This module has no access to target coordinates, IDs, contours or outcomes.
Unknown viewpoint directions remain categorical; they are never world axes.
"""

KINDS = {
    'near': 'near', 'by': 'near', 'beside': 'near', 'alongside': 'near',
    'next to': 'near', 'in': 'inside', 'within': 'inside', 'on': 'inside',
    'into': 'inside', 'onto': 'inside', 'outside': 'outside',
    'left of': 'left', 'right of': 'right', 'front of': 'front',
    'behind': 'behind', 'north of': 'north', 'south of': 'south',
    'east of': 'east', 'west of': 'west',
}
INVERSE = {'left': 'right', 'right': 'left', 'front': 'behind',
           'behind': 'front', 'north': 'south', 'south': 'north',
           'east': 'west', 'west': 'east', 'near': 'near',
           'inside': 'contains'}


def relation_metadata(role):
    def abstain(reason):
        return {'active': False, 'kind': 'none', 'reason': reason,
                'frame': 'unspecified', 'verified_reference_instance': False}

    graph = role.get('parse', {})
    target = graph.get('target_token')
    if target is None:
        return abstain('missing_target')
    if graph.get('scope_uncertain', True):
        return abstain('uncertain_scope')
    if len(role.get('reference_phrases', [])) != 1:
        return abstain('requires_single_visual_reference_phrase')
    # Public map references have separate geometry inputs. Do not attach their
    # predicate to an unrelated ordinary visual phrase.
    public = {path['nodes'][1] for path in graph.get('paths', [])
              if path.get('hops') == 1 and len(path.get('nodes', [])) == 2
              and path['nodes'][0] == target}
    edges = [edge for edge in graph.get('edges', [])
             if target in (edge['a'], edge['b'])
             and (edge['b'] if edge['a'] == target else edge['a']) not in public]
    refs = {edge['b'] if edge['a'] == target else edge['a'] for edge in edges}
    if len(refs) != 1:
        return abstain('missing_or_multiple_reference_nodes')
    kinds = set()
    for edge in edges:
        kind = KINDS.get(edge['relation'])
        if kind is None:
            return abstain('unsupported_relation')
        if edge['b'] == target:
            kind = INVERSE.get(kind)
        if kind is None:
            return abstain('unsupported_inverse')
        kinds.add(kind)
    if len(kinds) != 1:
        return abstain('multiple_relation_types')
    kind = next(iter(kinds))
    return {'active': True, 'kind': kind,
            'frame': 'world' if kind in {'north', 'south', 'east', 'west'} else 'unspecified',
            'reference_phrase': role['reference_phrases'][0],
            'reference_token': next(iter(refs)),
            'verified_reference_instance': False,
            'reason': 'single_scoped_dependency_proposal'}
