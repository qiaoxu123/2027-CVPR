"""All official TRAIN texts; no target IDs, category or coordinates are queried."""
import argparse,collections,gzip,json,random,time
from pathlib import Path
import spacy
from graphnav.checkpointing import file_sha256
from graphnav.structured_navigation import parse_instruction
from scripts.analyze_instruction_landmark_association import PublicNames,iter_train


def main():
    p=argparse.ArgumentParser()
    for k in ('annotations','catalog','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);public=PublicNames(json.loads(a.catalog.read_text()))
    tasks=[]
    for i,r in iter_train(a.annotations):
        for j,text in enumerate(r['descriptions']):tasks.append(dict(episode=i,instruction_index=j,text=text,map_name=f"{r['area']}_block_{r['block']}"))
    nlp=spacy.load('en_core_web_sm',disable=['ner']);records=[];start=time.perf_counter()
    for task,doc in zip(tasks,nlp.pipe((t['text'] for t in tasks),batch_size=128)):
        rec=parse_instruction(task['text'],task['map_name'],lambda _:doc,public)
        rec.update(episode=task['episode'],instruction_index=task['instruction_index']);records.append(rec)
        if len(records)%2000==0:print(json.dumps(dict(done=len(records),total=len(tasks),seconds=time.perf_counter()-start)),flush=True)
    with gzip.open(a.output/'structured_train.json.gz','wt',encoding='utf-8') as f:json.dump(records,f,ensure_ascii=False,separators=(',',':'))
    categories=collections.Counter(r['target']['category'] or 'unknown' for r in records)
    summary=dict(instructions=len(records),episodes=len({r['episode'] for r in records}),
        named_landmark=sum(bool(r['landmarks']) for r in records),multiple_landmarks=sum(len(r['landmarks'])>1 for r in records),
        target_role_resolved=sum(r['target']['entity'] is not None for r in records),target_category=dict(categories),
        target_attributes=sum(bool(r['target']['attributes']) for r in records),
        direct=sum(bool(r['direct_relations']) for r in records),indirect=sum(bool(r['indirect_relations']) for r in records),
        direct_frame_edge_pass=sum(any(p['frame_and_edge_check_passed'] for p in r['direct_relations']) for r in records),
        indirect_frame_edge_pass=sum(any(p['frame_and_edge_check_passed'] for p in r['indirect_relations']) for r in records),
        both=sum(bool(r['direct_relations']) and bool(r['indirect_relations']) for r in records),
        neither=sum(not r['direct_relations'] and not r['indirect_relations'] for r in records),
        ambiguity=dict(collections.Counter(flag for r in records for flag in r['ambiguity'])),
        directional_relations=dict(collections.Counter(e['predicate'] for r in records for e in r['relations'])),
        seconds=time.perf_counter()-start,annotations_sha256=file_sha256(a.annotations),catalog_sha256=file_sha256(a.catalog),
        parser_version=nlp.meta['version'],spacy=spacy.__version__,GT_used=False,
        code_sha256={str(p):file_sha256(p) for p in (Path(__file__),Path('graphnav/structured_navigation.py'),Path('scripts/analyze_instruction_landmark_association.py'))},
        human_review_completed=False,warning='Automatic proposals, not gold relations; ambiguous outputs are not enabled as training supervision.')
    strata=collections.defaultdict(list)
    for r in records:
        key='indirect' if r['indirect_relations'] else 'direct' if r['direct_relations'] else 'unresolved'
        strata[key].append(r)
    rng=random.Random(20261104);review=[]
    for key,items in sorted(strata.items()):
        for r in rng.sample(items,min(20,len(items))):review.append(dict(stratum=key,record=r,human_review=None,review_notes=''))
    (a.output/'review_sample.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
