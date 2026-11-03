"""Public-language region requirements; no coordinates, IDs or target labels."""
import re
from collections import deque


def _region(node):
    lemma=node['lemma'].lower();words=set(re.findall('[a-z]+',node['phrase'].lower()))
    if lemma in {'building','house','warehouse','church','chapel','hall'}:return 'Building'
    if lemma in {'lot','area','parking','ground'} and 'parking' in words:return 'Parking'
    if lemma in {'ground','grass','lawn','field','garden'} and not words&{'floor','storey','story'}:return 'Ground'
    return None


def region_requirement(nodes,graph):
    target=graph.get('target_token')
    if target not in nodes:return None
    direct=_region(nodes[target])
    if direct:return dict(region=direct,path=[target],kind='target-region-type')
    if graph.get('scope_uncertain',True):return None
    if nodes[target]['lemma'].lower() not in {'car','vehicle','van','truck','bus','jeep','suv','automobile'}:return None
    adjacency={}
    for edge in graph.get('edges',[]):
        a,b=edge['a'],edge['b']
        if a not in nodes or b not in nodes:continue
        # Directed containment only. Never treat next-to/behind as inclusion.
        allowed=edge['relation'] in {'in','inside','within'}
        allowed|=(edge['relation']=='of' and nodes[a]['lemma'].lower() in {'row','column','line','section'} and nodes[b]['lemma'].lower() in {'car','vehicle'})
        if allowed:adjacency.setdefault(a,[]).append(b)
    queue=deque([[target]]);seen={target}
    while queue:
        path=queue.popleft()
        if len(path)>1 and _region(nodes[path[-1]])=='Parking':
            return dict(region='Parking',path=path,kind='target-in-intermediate-region')
        if len(path)>=4:continue
        for other in adjacency.get(path[-1],[]):
            if other not in seen:seen.add(other);queue.append(path+[other])
    return None
