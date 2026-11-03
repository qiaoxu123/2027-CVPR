"""Public-text-only dependency proposals for target/reference roles.

This is a small local syntax parser, not a claim of full LLM semantic reasoning.
No target category, instance identifier, coordinate or TRAIN label is accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import spacy
from scripts.analyze_instruction_landmark_association import PublicNames, classify


def role(doc,mentions):
    graph=classify(doc,mentions,None,())
    target_id=graph.get('target_token')
    if target_id is None:
        return dict(target_text=doc.text,context_text=doc.text,reference_text=doc.text,
                    fallback=True,parse=graph)
    token=doc[target_id]
    modifiers=[child for child in token.children if child.dep_ in ('det','amod','compound','nummod')]
    ids={token.i}
    for child in modifiers:
        ids.update(t.i for t in child.subtree if t.pos_ not in ('VERB','AUX'))
    target=doc[min(ids):max(ids)+1].text
    for prep in token.children:
        if prep.lower_ not in ('with','of'):continue
        objects=[c for c in prep.children if c.dep_=='pobj']
        if not objects:continue
        obj=objects[0]
        if obj.lemma_.lower() not in {'roof','window','chimney','door','panel','balcony','ground','grass','lawn'}:continue
        words=list(prep.subtree)
        if len(words)<=14 and not any(m['start']<=words[-1].idx<m['end'] for m in mentions):
            target+=' '+doc[min(t.i for t in words):max(t.i for t in words)+1].text
    return dict(target_text=target,context_text=doc.text,reference_text=doc.text,
                fallback=False,parse=graph)


def reference_phrases(doc,parsed,mentions):
    """Visual descriptions of intermediate objects; names remain in global text."""
    graph=parsed['parse'];target=graph.get('target_token');ids=set()
    for path in graph.get('paths',[]):ids.update(path['nodes'][1:-1])
    for edge in graph.get('edges',[]):
        if target in (edge['a'],edge['b']):ids.add(edge['b'] if edge['a']==target else edge['a'])
    def is_public(t):return any(m['start']<=t.idx<m['end'] for m in mentions)
    chunks={chunk.root.i:chunk.text for chunk in doc.noun_chunks}
    phrases=[]
    for i in sorted(ids):
        token=doc[i]
        if i==target or is_public(token) or token.pos_ not in ('NOUN','PROPN'):continue
        text=chunks.get(i,token.text)
        for prep in token.children:
            if prep.lower_=='of':
                obj=next((t for t in prep.children if t.dep_=='pobj'),None)
                if obj is not None and obj.lemma_.lower() in ('car','vehicle','row','lot','tree'):
                    text+=' of '+chunks.get(obj.i,obj.text)
        phrases.append(text)
    if not phrases:
        for m in mentions:
            for kind in m['types']:
                visual_type='road' if 'road' in kind.lower() else 'building' if kind=='Building' else 'parking area' if kind=='Parking' else 'ground area' if kind=='Ground' else None
                if visual_type:phrases.append(visual_type)
    return list(dict.fromkeys(phrases))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-frames',type=Path,required=True)
    p.add_argument('--public-catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-phrases',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve earlier role cache')
    frames=[json.loads(line) for line in a.public_frames.read_text().splitlines()]
    queries={f['task_key']:(f['map_name'],f['instruction']) for f in frames}
    catalog=PublicNames(json.loads(a.public_catalog.read_text()))
    nlp=spacy.load('en_core_web_sm')
    result={}
    for key,(name,text) in queries.items():
        doc=nlp(text);mentions=catalog.match(name,text);parsed=role(doc,mentions)
        if a.reference_phrases:
            phrases=reference_phrases(doc,parsed,mentions)
            parsed.update(reference_phrases=phrases,reference_text='; '.join(phrases) if phrases else text,
                          original_instruction=text,reference_frame='unspecified unless explicit cardinal',version='local-roles-v2')
        result[key]=parsed
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    provenance=dict(public_frames_sha256=hashlib.sha256(a.public_frames.read_bytes()).hexdigest(),
                    public_catalog_sha256=hashlib.sha256(a.public_catalog.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    spacy_version=spacy.__version__, model_version=nlp.meta['version'],
                    tasks=len(result),fallback=sum(r['fallback'] for r in result.values()),
                    private_label_input=False,warning='Dependency proposals, not verified semantic gold.')
    a.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance))


if __name__=='__main__':main()
