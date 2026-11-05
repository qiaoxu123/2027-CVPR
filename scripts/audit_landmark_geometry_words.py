"""Whole TRAIN parser-only audit. Original descriptions never go to BERT."""
import argparse,json,collections
from pathlib import Path
from graphnav.checkpointing import atomic_json
from scripts.train_landmark_geometry_words import LandmarkWordsDataset


def main():
    p=argparse.ArgumentParser()
    for k in ('train-jsonl','public-objects','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);ds=LandmarkWordsDataset(a.train_jsonl,a.public_objects);raw=[json.loads(x) for x in a.train_jsonl.read_text().splitlines()];assert len(ds)==21878
    dist=collections.defaultdict(collections.Counter);phrases=collections.Counter();examples=[];seen=collections.Counter();texts=set()
    with (a.output/'extraction_audit.jsonl').open('w') as f:
        for i,r in enumerate(ds.records):
            split='fit' if r['map_name'].startswith('cambridge') else 'dev';dist[split][r['status']]+=1
            pairs=[dict(name=x['name'],geometry_words=x['geometry_words'],public_id=x['public_id']) for x in r['anchors']]
            unit={(x['name'],x['geometry_words']) for x in r['anchors']}
            for name,word in unit:
                texts.add(name)
                if word:phrases[word]+=1;texts.add(word)
            row=dict(index=i,map_name=r['map_name'],status=r['status'],original_instruction=raw[i]['instruction'],model_text_pairs=pairs)
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if seen[r['status']]<8:examples.append(row);seen[r['status']]+=1
    atomic_json(dict(n=len(ds),counts=dict(ds.counter),split={k:dict(v) for k,v in dist.items()},unique_model_texts=len(texts),phrases=dict(phrases.most_common()),GT_input=False,scope='lexically adjacent, not verified target-to-landmark directness'),a.output/'coverage.json');atomic_json(examples,a.output/'examples.json');atomic_json(sorted(texts),a.output/'allowed_BERT_texts.json');print(json.dumps(dict(ds.counter)),flush=True)

if __name__=='__main__':main()
