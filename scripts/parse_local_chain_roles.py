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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-frames',type=Path,required=True)
    p.add_argument('--public-catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve earlier role cache')
    frames=[json.loads(line) for line in a.public_frames.read_text().splitlines()]
    queries={f['task_key']:(f['map_name'],f['instruction']) for f in frames}
    catalog=PublicNames(json.loads(a.public_catalog.read_text()))
    nlp=spacy.load('en_core_web_sm')
    result={key:role(nlp(text),catalog.match(name,text)) for key,(name,text) in queries.items()}
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
