"""Conservative binary between hyperedges from public-language role proposals.

A pair is not two independent near relations. Quantity-two references are not
fabricated as distinct instances. This codec accepts no world target labels.
"""
from collections import defaultdict, deque


def scoped_between(nodes, graph, public_nodes):
    target=graph.get('target_token')
    if target not in nodes or graph.get('scope_uncertain',True):return []
    paired=defaultdict(set);containment=defaultdict(list)
    for e in graph.get('edges',[]):
        if e['a'] not in nodes or e['b'] not in nodes:continue
        if e['relation']=='between':paired[e['a']].add(e['b'])
        if e['relation'] in {'in','inside','within'}:containment[e['a']].append(e['b'])
    paths={target:[target]};queue=deque([target])
    while queue:
        node=queue.popleft()
        if len(paths[node])>=4:continue
        for other in containment[node]:
            if other not in paths:paths[other]=paths[node]+[other];queue.append(other)
    output=[]
    for owner,refs in sorted(paired.items()):
        if owner not in paths or len(refs)!=2:continue
        refs=sorted(refs)
        endpoints=[dict(token=i,phrase=nodes[i]['phrase'],public_ids=sorted(set(public_nodes.get(i,[]))), name_unresolved=bool(nodes[i].get('proper_name_hint') and not public_nodes.get(i))) for i in refs]
        if endpoints[0]['public_ids'] and endpoints[0]['public_ids']==endpoints[1]['public_ids']:continue
        output.append(dict(owner_token=owner,owner_phrase=nodes[owner]['phrase'],
            target_to_owner=paths[owner],endpoints=endpoints,
            kind='unresolved-name-endpoint' if any(x['name_unresolved'] for x in endpoints) else
                 'two-public-anchors' if all(x['public_ids'] for x in endpoints) else
                 'one-public-one-visual' if any(x['public_ids'] for x in endpoints) else 'two-visual-references',
            reference_frame='orientation-independent-between',semantic_truth_verified=False))
    return output
