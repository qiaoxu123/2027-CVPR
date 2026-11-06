"""GT-free dependency roles and directed language paths; no world-frame rules.

The legacy audit classifier can use GT category/ID. This inference wrapper
intentionally exposes neither argument and supplies None/empty identity only.
"""
import re
from scripts.analyze_instruction_landmark_association import classify,TYPES
from scripts.prepare_unified_v2_landmark_relation_ablation import _ATTACHED


def parse_doc(doc,text,map_name,matcher):
    mentions=matcher.match(map_name,text)
    # Sentence fragments can make a later noun ROOT. An explicit initial
    # described object is a text-only hint; never a category from GT.
    hint=None
    candidates=set().union(*TYPES.values())
    for token in doc:
        if token.lemma_.lower() not in candidates or any(m['start']<=token.idx<m['end'] for m in mentions):continue
        prefix=text[:token.idx].casefold().strip()
        prefix=re.sub(r'^(?:navigate|go|fly|find|locate|reach)\s+(?:to\s+)?','',prefix)
        if len(prefix.split())<=12 and re.match(r'^(?:the|a|an|one|this|that)\b',prefix) and not re.search(r'\b(?:of|in|on|behind|beside|near|from|with|at|to)\b',prefix):hint=token.idx
        break
    result=classify(doc,mentions,None,(),target_hint_char=hint)
    chunks=list(doc.noun_chunks)
    def entity(i):
        token=doc[i];named=[m for m in mentions if m['start']<=token.idx<m['end']]
        if named:
            m=named[0];start,end=m['start'],m['end'];ids=m['public_ids'];kind='named_landmark'
        else:
            spans=[c for c in chunks if c.start<=i<c.end]
            span=min(spans,key=lambda c:len(c)) if spans else doc[i:i+1]
            start,end=span.start_char,span.end_char;ids=[];kind='ordinary_object'
        return dict(node=i,text=text[start:end],start=start,end=end,public_ids=ids,kind=kind)
    ids=set()
    for edge in result.get('edges',[]):ids.update([edge['a'],edge['b']])
    for p in result.get('paths',[]):ids.update(p['nodes'])
    entities={i:entity(i) for i in ids}
    target_id=result.get('target_token_index',next((p['nodes'][0] for p in result.get('paths',[])),None))
    if target_id is None and result.get('target'):
        matches=[t.i for t in doc if t.text==result['target']]
        if len(matches)==1:target_id=matches[0];entities[target_id]=entity(target_id)
    target=entities.get(target_id)
    if target:target={**target,'role':'target'}
    paths=[]
    for path in result.get('paths',[]):
        steps=[]
        for j,relation in enumerate(path['relations']):
            x,y=path['nodes'][j:j+2]
            edge=next(e for e in result['edges'] if {e['a'],e['b']}=={x,y} and e['relation']==relation)
            # Direction explicitly describes grammatical subject->reference,
            # independent of traversal and independent of any GT map direction.
            orientation=1 if edge['a']==x else -1
            recovered=_ATTACHED.search(text[:entities[edge['b']]['start']])
            original_relation=relation
            if recovered:relation=recovered.group(1).strip()
            unresolved=any(w in relation for w in ('left','right','front','behind','back','ahead'))
            steps.append(dict(subject=entities[edge['a']],object=entities[edge['b']],relation=relation,orientation=orientation,reference_frame='unresolved_relative' if unresolved else 'lexical_not_geographically_calibrated',original_relation=original_relation,evidence=edge['evidence']))
        end=entities[path['nodes'][-1]]
        paths.append(dict(semantic_confident=all(s['relation'] not in {'to','of','with','possessive region association','associated region'} for s in steps),nodes=[entities[i] for i in path['nodes']],steps=steps,hops=path['hops'],public_ids=end['public_ids'],public_names=path['public_names']))
    category=result['category']
    if paths and not any(p['semantic_confident'] for p in paths):category='uncertain'
    return dict(raw_category=result['category'],has_direct=any(p['hops']==1 for p in paths),has_indirect=any(p['hops']>1 for p in paths),instruction=text,map_name=map_name,category=category,target_span=target,entities=list(entities.values()),edges=result.get('edges',[]),paths=paths,scope_uncertain=bool(result.get('scope_uncertain',True)),scope_repairs=result.get('scope_repairs',[]),GT_used=False,parser='spacy_dependency_GT_free_v1')


def parse_instruction(nlp,text,map_name,matcher):
    return parse_doc(nlp(text),text,map_name,matcher)
