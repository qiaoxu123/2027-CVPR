"""Public text-only scoped relation proposals for a bounded local factor.

Multiple references remain distinct; map anchors belong to their actual source
node. These are dependency proposals, never reference-instance ground truth.
"""
from collections import defaultdict

RELATIONS = ('unknown', 'near', 'inside', 'contains', 'on', 'supports',
             'left', 'right', 'front', 'behind', 'north', 'south', 'east',
             'west', 'outside', 'associated', 'identity')
RELATION_IDS = {kind: i for i, kind in enumerate(RELATIONS)}
RAW_KINDS = {
    'near': 'near', 'by': 'near', 'beside': 'near', 'next to': 'near',
    'alongside': 'near', 'at': 'near', 'in': 'inside', 'inside': 'inside',
    'within': 'inside', 'into': 'inside', 'on': 'on', 'onto': 'on',
    'left of': 'left', 'right of': 'right', 'front of': 'front',
    'behind': 'behind', 'north of': 'north', 'south of': 'south',
    'east of': 'east', 'west of': 'west', 'outside': 'outside',
    'associated region': 'associated', 'possessive region association': 'associated',
}
INVERSE = {'near': 'near', 'inside': 'contains', 'contains': 'inside',
           'on': 'supports', 'supports': 'on', 'left': 'right', 'right': 'left',
           'front': 'behind', 'behind': 'front', 'north': 'south', 'south': 'north',
           'east': 'west', 'west': 'east', 'associated': 'associated'}


def _frame(kind):
    return 'world' if kind in {'north', 'south', 'east', 'west'} else 'unspecified'


def build_relation_roles(doc, parsed, mentions, capacity=8):
    if capacity < 1:
        raise ValueError('positive capacity required')
    graph = parsed.get('parse', {})
    target = graph.get('target_token')
    result = dict(target_text=parsed['target_text'], original_instruction=doc.text,
                  references=[], unsupported_edges=[], dropped_capacity=0,
                  global_scope_uncertain=graph.get('scope_uncertain', True),
                  version='local-relation-factor-v1', verified_reference_instances=False)
    if target is None or result['global_scope_uncertain']:
        return result
    public = {}
    for mention in mentions:
        token = next((t for t in doc if t.idx <= mention['start'] < t.idx+len(t.text)), None)
        if token is not None:
            public[token.i] = mention
    chunks = {chunk.root.i: chunk.text for chunk in doc.noun_chunks}

    def phrase(node):
        text = chunks.get(node, doc[node].text)
        # Preserve reference group identity such as "fourth row of cars".
        for prep in doc[node].children:
            if prep.lower_ == 'of':
                obj = next((c for c in prep.children if c.dep_ == 'pobj'), None)
                if obj is not None and obj.lemma_.lower() in {'car', 'vehicle', 'row', 'lot', 'tree'}:
                    text += ' of '+chunks.get(obj.i, obj.text)
        return text

    adjacent = defaultdict(list)
    for edge in graph.get('edges', []):
        kind = RAW_KINDS.get(edge['relation'])
        if kind is None:
            result['unsupported_edges'].append(dict(a=edge['a'], b=edge['b'], raw_relation=edge['relation']))
            continue
        adjacent[edge['a']].append((edge['b'], kind, edge['relation']))
        if kind in INVERSE:
            adjacent[edge['b']].append((edge['a'], INVERSE[kind], edge['relation']))

    for node, kind, raw in sorted(set(adjacent[target])):
        if len(result['references']) >= capacity:
            result['dropped_capacity'] += 1
            continue
        if not 0 <= node < len(doc):
            raise ValueError('proposal token out of range')
        anchors = []
        if node in public:
            anchors.append(dict(kind='identity', public_ids=public[node]['public_ids'],
                                names=public[node]['public_names']))
        else:
            # One intermediate region->named-map link. Do not jump through the
            # target to collect unrelated anchors or collapse a longer chain.
            for other, anchor_kind, _ in sorted(set(adjacent[node])):
                if other != target and other in public:
                    anchors.append(dict(kind=anchor_kind, public_ids=public[other]['public_ids'],
                                        names=public[other]['public_names']))
        result['references'].append(dict(token=node, phrase=phrase(node), kind=kind,
            raw_relation=raw, frame=_frame(kind), public_ids=public.get(node, {}).get('public_ids', []),
            anchors=anchors))
    return result
