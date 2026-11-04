"""Text/public-name-only structured language; no target annotation API."""
import re
from collections import defaultdict,deque
from scripts.analyze_instruction_landmark_association import classify,TYPES

COLORS={'white','black','grey','gray','red','blue','green','brown','yellow','orange','silver','beige','tan','pink','purple'}
SHAPES={'square','rectangular','round','circular','triangular','flat','sloped','pointed','l-shaped','u-shaped'}
SIZES={'large','small','big','tiny','tall','short','long','wide','narrow','largest','smallest','tallest'}


def relation_frame(relation):
    words=set(re.findall(r'[a-z]+',relation.lower()))
    if words&{'north','south','east','west','northeast','northwest','southeast','southwest'}:
        return dict(frame='world_cardinal',requires_review=False)
    if words&{'front','behind','back','rear','left','right','above','below','top','bottom'}:
        return dict(frame='reference_frame_unknown',requires_review=True)
    return dict(frame='topological' if words&{'in','inside','within','contain','outside','on'} else 'metric_or_adjacency',requires_review=False)


def parse_instruction(text,map_name,nlp,public_names):
    mentions=public_names.match(map_name,text)
    # A catalog road name followed by a generic 'Road' is one referent, not
    # an extra object/hop. Keep the exact catalog span as separate provenance.
    scoped=[]
    for m in mentions:
        m=dict(m);m['catalog_span']=[m['start'],m['end']]
        suffix=re.match(r'\s+(Road|Street|Footpath|Building)\b',text[m['end']:],re.I)
        if suffix and ((suffix[1].lower()=='building' and m['types']==['Building']) or
                       (suffix[1].lower()!='building' and set(m['types'])<={'TrafficRoad','Footpath','Bridge'})):
            m['end']+=suffix.end();m['text']=text[m['start']:m['end']]
        scoped.append(m)
    mentions=scoped
    # Crucial: the old audit used annotated target type/identity. This call uses
    # the complete lexical category set and NO private target information.
    doc=nlp(text)
    parsed=classify(doc,mentions,'',())
    target_id=parsed.get('target_token')
    lexical_fallback=False
    if target_id is None and parsed.get('target'):
        candidate=next((t for t in doc if t.text==parsed['target'] and t.lemma_.lower() in set().union(*TYPES.values())),None)
        if candidate is not None:target_id=candidate.i;lexical_fallback=True
    named_target=None
    for m in mentions:
        prefix=text[:m['start']].strip().lower()
        suffix=text[m['end']:].strip(' .!')
        if not suffix and re.fullmatch(r'(?:(?:navigate|go|fly|head|travel) to|find|locate|reach|the)?',prefix):named_target=m
    ids={e[k] for e in parsed.get('edges',[]) for k in ('a','b')}
    if target_id is not None:ids.add(target_id)
    for m in mentions:
        ids.update(t.i for t in doc if t.idx<=m['start']<t.idx+len(t.text))
    if target_id is None and named_target:
        target_id=next((t.i for t in doc if t.idx<=named_target['start']<t.idx+len(t.text)),None)
        if target_id is not None:ids.add(target_id)
    entities=[]
    for t in doc:
        if t.i not in ids:continue
        mention=next((m for m in mentions if t.idx<=m['start']<t.idx+len(t.text)),None)
        entities.append(dict(id=f'e{t.i}',text=t.text,span=[t.idx,t.idx+len(t.text)],
            role='target' if t.i==target_id else 'landmark' if mention else 'reference_object',
            public_names=mention['public_names'] if mention else []))
    target=doc[target_id] if target_id is not None else None
    categories=[c for c,words in TYPES.items() if target is not None and target.lemma_.lower() in words]
    if named_target:
        categories=list(named_target['types'])
    attributes=[]
    if target is not None:
        attached=[t for t in doc if t.head.i==target.i and t.dep_ in ('amod','compound','nummod')]
        # Target-owned roof/wall phrases, not descriptors of other objects.
        for token in doc:
            if token.lemma_.lower() in {'roof','wall','window','door'}:
                ancestors=list(token.ancestors)
                intervening=[x for x in ancestors if x.pos_ in ('NOUN','PROPN') and x.i!=target_id]
                if any(x.i==target_id for x in ancestors) and not intervening:
                    if token.dep_=='pobj' and token.head.lower_=='with':
                        attached += [t for t in token.children if t.dep_ in ('amod','compound','nummod')]+[token]
        # Compound roof attributes ('grey roof building') also have local modifiers.
        attached += [child for token in list(attached) if token.lemma_.lower() in {'roof','wall','window','door'} for child in token.children if child.dep_ in ('amod','compound','nummod')]
        attached=list({t.i:t for t in attached}.values())
        for t in attached:
            value=t.text.lower();kind='color' if value in COLORS else 'shape' if value in SHAPES else 'size' if value in SIZES else 'modifier'
            attributes.append(dict(kind=kind,value=t.text,span=[t.idx,t.idx+len(t.text)],entity=f'e{target_id}'))
    relations=[]
    for j,e in enumerate(parsed.get('edges',[])):
        aa,bb=doc[e['a']],doc[e['b']];start=min(aa.idx,bb.idx);end=max(aa.idx+len(aa.text),bb.idx+len(bb.text))
        rel=e['relation']
        head=e['evidence'].split(' -> ')[0].lower()
        if rel in ('to','from') and head in ('next','across','adjacent','close'):
            rel=head+' '+rel
        if rel=='by' and head.startswith(('surround','enclos','border')):rel=head+' by'
        if 'side' in rel:
            direction=re.findall(r'\b(left|right|front|back|rear|north|south|east|west)\s+(?:hand\s+)?side\b',text[start:end],re.I)
            if direction and direction[-1].lower() not in rel.split():rel=direction[-1].lower()+' '+rel
        frame=relation_frame(rel)
        relations.append(dict(id=f'r{j}',dependency_predicate=e['relation'],subject=f'e{e["a"]}',object=f'e{e["b"]}',predicate=rel,
            source_span=[start,end],source_text=text[start:end],dependency_evidence=e['evidence'],
            direction='subject_relative_to_object',reference_frame=frame,
            relation_kind='association' if rel in ('with','possessive region association','associated region','of') else 'scope_unresolved' if rel in ('to','from','for') else 'spatial_proposal',
            ambiguity=['reference_frame_unresolved'] if frame['requires_review'] else [],
            supervision_status='parser_proposal_not_verified'))
    # Rebuild complete shortest chains; the historical audit cut off at 4 hops.
    graph=defaultdict(list)
    for edge in parsed.get('edges',[]):
        graph[edge['a']].append((edge['b'],edge['relation']));graph[edge['b']].append((edge['a'],edge['relation']))
    public_by_token={t.i:m for t in doc for m in mentions if t.idx<=m['start']<t.idx+len(t.text)}
    complete=[]
    if target_id is not None:
        queue=deque([(target_id,[target_id],[])]);visited={target_id}
        while queue:
            node,nodes,rels=queue.popleft()
            if node!=target_id and node in public_by_token:
                complete.append(dict(nodes=nodes,relations=rels,hops=len(rels),public_names=public_by_token[node]['public_names']))
            for nxt,rel in graph[node]:
                if nxt not in visited:visited.add(nxt);queue.append((nxt,nodes+[nxt],rels+[rel]))
    paths=[]
    for path in complete:
        chain=[]
        for a,b,rel in zip(path['nodes'],path['nodes'][1:],path['relations']):
            edge=next((e for e in relations if {e['subject'],e['object']}=={f'e{a}',f'e{b}'} and e['dependency_predicate']==rel),None)
            if edge:chain.append(dict(relation_id=edge['id'],traversal='forward' if edge['subject']==f'e{a}' else 'reverse'))
        paths.append(dict(entities=[f'e{x}' for x in path['nodes']],relations=chain,landmark_names=path['public_names'],hops=path['hops'],
            ambiguity=sorted({flag for e in relations if e['id'] in {x['relation_id'] for x in chain} for flag in e['ambiguity']}),
            frame_and_edge_check_passed=all(e['relation_kind']=='spatial_proposal' and not e['ambiguity'] for e in relations if e['id'] in {x['relation_id'] for x in chain}) and all(x['traversal']=='forward' for x in chain)))
    ambiguity=[]
    if target is not None:
        other=[t for t in doc if t.i<target_id and t.lemma_.lower() in set().union(*TYPES.values()) and not any(t.idx<=m['start']<t.idx+len(t.text) for m in mentions)]
        if other:ambiguity.append('earlier_target_category_noun_scope_requires_review')
    if lexical_fallback:ambiguity.append('target_lexical_fallback_scope_uncertain')
    if target is None:ambiguity.append('target_role_unresolved')
    if len(categories)!=1:ambiguity.append('target_category_ambiguous_or_unknown')
    if parsed.get('scope_uncertain'):ambiguity.append('relation_scope_unresolved')
    if any(len(m['public_names'])>1 for m in mentions):ambiguity.append('public_alias_multiple_names')
    if re.search(r'\b(?:it|they|this|that|which)\b',text,re.I):ambiguity.append('coreference_parser_assumption')
    if any(e['reference_frame']['requires_review'] for e in relations):ambiguity.append('reference_frame_unresolved')
    if any(x['traversal']=='reverse' for p in paths for x in p['relations']):ambiguity.append('chain_contains_reverse_relation')
    relation_groups=[]
    grouped=defaultdict(list)
    for e in relations:
        if 'between' in e['predicate']:grouped[(e['subject'],e['predicate'])].append(e)
    for (subject,predicate),edges in grouped.items():
        relation_groups.append(dict(subject=subject,predicate=predicate,relation_ids=[e['id'] for e in edges],
            reference_entities=[e['object'] for e in edges],requires_joint_interpretation=True,
            ambiguity=[] if len(edges)>=2 else ['between_missing_second_reference']))
    unbound=[]
    for chunk in doc.noun_chunks:
        if any(t.pos_=='PROPN' for t in chunk) and not any(chunk.start_char<m['end'] and chunk.end_char>m['start'] for m in mentions):
            unbound.append(dict(text=chunk.text,span=[chunk.start_char,chunk.end_char],binding='unresolved_not_in_public_catalog'))
    if unbound:ambiguity.append('unbound_proper_name_candidate')
    for entity in entities:
        token=doc[int(entity['id'][1:])]
        entity['attributes']=[dict(text=t.text,span=[t.idx,t.idx+len(t.text)]) for t in token.children if t.dep_ in ('amod','compound','nummod')]
    return dict(original_text=text,map_name=map_name,landmarks=mentions,relation_groups=relation_groups,
        unbound_landmark_candidates=unbound,approved_for_training=False,
        target=dict(entity=f'e{target_id}' if target_id is not None else None,name=named_target['text'] if named_target else None,
            category=categories[0] if len(categories)==1 else None,category_candidates=categories,attributes=attributes),
        entities=entities,relations=relations,direct_relations=[p for p in paths if p['hops']==1],
        indirect_relations=[p for p in paths if p['hops']>1],ambiguity=ambiguity,
        scope_repairs=parsed.get('scope_repairs',[]),parser_status='automatic_unverified',
        target_GT_used=False,normal_inference_enabled=False)
