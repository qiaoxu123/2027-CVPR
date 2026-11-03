"""Export public-text-only target/intermediate region proposals, not gold roles."""
import argparse
import json
from pathlib import Path
import hashlib
import spacy
from graphnav.region_chain_requirement import region_requirement
from scripts.analyze_instruction_landmark_association import PublicNames
from scripts.parse_local_chain_roles import role


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('public-frames','catalog','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous proposals')
    queries={r['task_key']:(r['map_name'],r['instruction']) for r in map(json.loads,a.public_frames.read_text().splitlines())}
    names=PublicNames(json.loads(a.catalog.read_text()));nlp=spacy.load('en_core_web_sm');output={}
    for key,(map_name,text) in queries.items():
        doc=nlp(text);parsed=role(doc,names.match(map_name,text));chunks={x.root.i:x.text for x in doc.noun_chunks}
        nodes={t.i:dict(lemma=t.lemma_.lower(),phrase=chunks.get(t.i,t.text)) for t in doc}
        proposal=region_requirement(nodes,parsed['parse']) if not parsed['fallback'] else None
        if proposal:proposal['path_phrases']=[nodes[i]['phrase'] for i in proposal['path']]
        output[key]=dict(map_name=map_name,instruction=text,proposal=proposal,scope_uncertain=parsed['parse'].get('scope_uncertain',True))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2,ensure_ascii=False))
    a.output.with_suffix('.provenance.json').write_text(json.dumps(dict(queries=len(output),accepted=sum(x['proposal'] is not None for x in output.values()),
        private_labels_read=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        proposal_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),warning='Conservative dependency proposals; not verified semantic truth'),indent=2))


if __name__=='__main__':main()
