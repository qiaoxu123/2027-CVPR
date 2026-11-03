"""Frozen text stratification of existing original-paper Val-Seen rollouts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from scripts.analyze_instruction_landmark_association import (
    PublicNames, classify, iter_train, GENERIC_NAMES, HELPERS, TYPES)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def aggregate(rows):
    if not rows:
        return dict(episodes=0)
    if len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate evaluation episode')
    if any(r['metrics']['SR'] not in (0, 1) for r in rows):
        raise ValueError('SR must be an episode binary outcome')
    n=len(rows); successes=sum(r['metrics']['SR'] for r in rows); p=successes/n
    z=1.959963984540054; denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return dict(episodes=n,successes=int(successes),SR=p,
        **{k:statistics.mean(r['metrics'][k] for r in rows) for k in ('OSR','SPL','NE','nDTW')},
        sr_wilson95=[center-half,center+half],
        maps=len({r['map_name'] for r in rows}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('annotations','catalog','evaluation','output'):
        parser.add_argument('--'+k,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.annotations.name!='citynav_val_seen.json':
        raise ValueError('Only requested existing Val-Seen analysis supported')
    import spacy
    catalog=json.loads(args.catalog.read_text()); names=PublicNames(catalog)
    started=json.loads((args.evaluation/'eval_started.json').read_text())
    published=json.loads((args.evaluation/'metrics.json').read_text())
    if started['args']['model']!='graphnav-v1' or started['args']['stage1_visual_control']!='real':
        raise ValueError('Original real-RGB GraphNav-V1 evaluation required')
    predictions={}
    for line in (args.evaluation/'predictions.jsonl').open():
        r=json.loads(line)
        if r['index'] in predictions:
            raise ValueError('Duplicate stored episode')
        predictions[r['index']]={k:r[k] for k in ('index','episode_id','map_name','metrics')}
        predictions[r['index']]['start_xyz']=r['poses'][0][:3]
    raw=list(iter_train(args.annotations))
    if len(raw)!=2470 or set(predictions)!=set(range(len(raw))):
        raise ValueError('Complete 2470/2470 matching episode set required')
    all_metrics=aggregate(list(predictions.values()))
    for key in ('SR','OSR','SPL','NE','nDTW'):
        if not math.isclose(all_metrics[key],published['metrics'][key],rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('Stored aggregate differs: '+key)
    # No classifier revision or threshold tuning: exact existing TRAIN recipe.
    nlp=spacy.load('en_core_web_sm',disable=['ner'])
    results=[]
    for (i, task), doc in zip(raw,nlp.pipe((t['descriptions'][0] for _,t in raw),batch_size=128)):
        mp=f"{task['area']}_block_{task['block']}"; text=task['descriptions'][0]
        pred=predictions[i]
        if pred['episode_id']!=[mp,task['object_ids'][0],task['ann_ids'][0]] or pred['map_name']!=mp:
            raise ValueError('Episode identity mismatch: '+str(i))
        if any(abs(a-b)>1e-5 for a,b in zip(pred['start_xyz'],task['trajectory'][0][:3])):
            raise ValueError('Initial XYZ mismatch: '+str(i))
        category=catalog[mp][str(task['object_ids'][0])]['object_type']
        mentions=names.match(mp,text)
        parsed=classify(doc,mentions,category,identity_ids=[str(x) for x in task['object_ids']])
        ignored=GENERIC_NAMES|HELPERS|set().union(*TYPES.values())|{'black','white','gray','grey','blue','red','green','brown','beige','yellow','orange','silver','tan'}
        cues=[c.text for c in doc.noun_chunks if any(t.pos_=='PROPN' and t.text.lower() not in ignored and len(t.text)>1 for t in c)] if not mentions else []
        if parsed['category']=='no_confirmed_anchor' and cues:
            parsed['category']='uncertain'
        results.append(dict(index=i,episode_id=pred['episode_id'],map_name=mp,instruction=text,
            target_category=category,relation_class=parsed['category'],paths=parsed.get('paths',[]),
            metrics=pred['metrics']))
    groups={c:aggregate([r for r in results if r['relation_class']==c]) for c in sorted({r['relation_class'] for r in results})}
    direct=[r for r in results if r['relation_class']=='direct_geometry']
    by_target={c:aggregate([r for r in direct if r['target_category']==c]) for c in sorted({r['target_category'] for r in direct})}
    # Map-cluster bootstrap: repeated episodes in one map are not independent.
    clusters=defaultdict(list)
    for r in direct:
        clusters[r['map_name']].append(r['metrics']['SR'])
    rng=random.Random(0); keys=sorted(clusters); draws=[]
    for _ in range(2000):
        sample=[x for k in rng.choices(keys,k=len(keys)) for x in clusters[k]]
        draws.append(statistics.mean(sample))
    draws.sort()
    args.output.mkdir(parents=True)
    summary=dict(all=all_metrics,by_relation=groups,direct_by_target=by_target,
        direct_map_bootstrap95=[draws[49],draws[1949]],
        model=published['model'],checkpoint_sha256=started['checkpoint_sha256'],
        protocol=started['args'],classifier='Unmodified TRAIN association classifier, automatic not human gold',
        annotation_usage='Posthoc target category and explicit identity only; no GT coordinates used in relation classification or model rerun',
        split='val_seen',coverage='2470/2470; existing rollouts, no new navigation, no tuning',
        parser=dict(spacy=spacy.__version__,name=nlp.meta['name'],version=nlp.meta['version']),
        source_sha256={str(p):sha(p) for p in (args.annotations,args.catalog,
            args.evaluation/'eval_started.json',args.evaluation/'metrics.json',args.evaluation/'predictions.jsonl',
            Path('scripts/analyze_instruction_landmark_association.py'),Path(__file__))})
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (args.output/'episodes.jsonl').open('w') as f:
        for r in results:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('all','by_relation','direct_by_target','direct_map_bootstrap95')},indent=2),flush=True)


if __name__=='__main__':
    main()
