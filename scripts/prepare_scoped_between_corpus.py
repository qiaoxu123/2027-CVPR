"""TRAIN public-text-only between supervision proposals; no target metadata use."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import spacy
from graphnav.scoped_between import scoped_between
from scripts.analyze_instruction_landmark_association import PublicNames,iter_train
from scripts.parse_local_chain_roles import role


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    # Explicit whitelist: never read object_ids, marker/target positions or contours.
    inputs=[dict(episode=i,map_name=f"{r['area']}_block_{r['block']}",instruction=r['descriptions'][0]) for i,r in iter_train(a.annotations)]
    names=PublicNames(json.loads(a.catalog.read_text()));nlp=spacy.load('en_core_web_sm');rows=[]
    for index,(r,doc) in enumerate(zip(inputs,nlp.pipe([r['instruction'] for r in inputs],batch_size=128))):
        mentions=names.match(r['map_name'],r['instruction']);parsed=role(doc,mentions)
        chunks={c.root.i:c for c in doc.noun_chunks}
        nodes={}
        for t in doc:
            chunk=chunks.get(t.i);phrase=chunk.text if chunk is not None else t.text
            visual_heads={'car','vehicle','van','truck','bus','tree','space','spot','row','column','line'}
            hint=bool(chunk is not None and any(w.pos_=='PROPN' for w in chunk) and t.lemma_.lower() not in visual_heads)
            # Preserve row/line composition rather than reducing 'row of trees' to 'row'.
            if t.lemma_.lower() in {'row','line','column'}:
                for prep in t.children:
                    if prep.lower_!='of':continue
                    obj=next((w for w in prep.children if w.dep_=='pobj'),None)
                    if obj is not None:
                        part=chunks.get(obj.i);phrase+=' of '+(part.text if part is not None else obj.text)
            nodes[t.i]=dict(phrase=phrase,proper_name_hint=hint)
        public={t.i:m['public_ids'] for m in mentions for t in doc if t.idx<=m['start']<t.idx+len(t.text)}
        groups=scoped_between(nodes,parsed['parse'],public)
        rows.append(dict(**r,target_phrase=parsed['target_text'],groups=groups,
            scope_uncertain=parsed['parse'].get('scope_uncertain',True),
            binary_between_edges=sum(e['relation']=='between' for e in parsed['parse'].get('edges',[]))))
        if (index+1)%2000==0:print(json.dumps(dict(parsed=index+1,groups=sum(len(x['groups']) for x in rows))),flush=True)
    summary=dict(tasks=len(rows),tasks_with_proposals=sum(bool(r['groups']) for r in rows),
        groups_by_kind=dict(Counter(g['kind'] for r in rows for g in r['groups'])),
        tasks_with_between_edges=sum(r['binary_between_edges']>0 for r in rows),
        private_target_metadata_used=False,normal_inference_modified=False,
        warning='Dependency proposals, not verified semantic or pixel gold; binary pair requirement deliberately abstains on quantity-two and unresolved scopes.',
        spacy_version=spacy.__version__,parser_version=nlp.meta['version'],
        sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in dict(annotations=a.annotations,catalog=a.catalog,producer=Path(__file__),codec=Path('graphnav/scoped_between.py')).items()})
    a.output.mkdir(parents=True);(a.output/'records.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
