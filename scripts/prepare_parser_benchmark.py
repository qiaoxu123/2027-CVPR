"""Seeded query sampling; outcomes and map/GT stay out of model inputs."""
import argparse,json,collections,random,hashlib
from pathlib import Path
from graphnav.parser_benchmark import public_record

def main():
 p=argparse.ArgumentParser();p.add_argument('--bearing-run',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);records=json.load(open(a.manifest))['records'];catalog=json.load(open(a.catalog));parsed=[json.loads(s) for s in (a.bearing_run/'data/parsed.jsonl').read_text().splitlines()];rows=[json.loads(s) for s in (a.bearing_run/'confirmation/relations_angle_rows.jsonl').read_text().splitlines()];rng=random.Random(42);used=set();selected=[]
 pools={
 'large_error_context_absent':[x for x in rows if x['error']>90 and x['target_index_offline']<0],
 'large_error_context_present':[x for x in rows if x['error']>90 and x['target_index_offline']>=0],
 'correct_direction':[x for x in rows if x['top1']==1]}
 for group,pool in pools.items():
  shuffled=list(pool);rng.shuffle(shuffled);buckets=collections.defaultdict(list)
  for x in shuffled:
   category=parsed[x['index']]['category'];buckets[category].append(x)
  take=[]
  while len(take)<100:
   changed=False
   for category in sorted(buckets):
    while buckets[category]:
     x=buckets[category].pop()
     if x['index'] not in used:used.add(x['index']);take.append(dict(group=group,selected_anchor=x['anchor'],error=x['error'],target_context=x['target_index_offline']>=0,index=x['index']));changed=True;break
    if len(take)>=100:break
   if not changed:raise RuntimeError('Not enough unique queries')
  selected.extend(take)
 # Random disjoint remainder, uniform queries, independent of prediction outcomes.
 randomids=rng.sample([i for i in range(len(records)) if i not in used],100)
 selected.extend(dict(group='random_TRAIN_remainder',index=i) for i in randomids);public=[];private=[]
 for x in selected:
  i=x['index'];r=records[i];names=sorted({o['name'] for o in catalog[r['map_name']].values() if o.get('name')});public.append(public_record(dict(index=i,instruction=r['instruction'],landmark_catalog=names)));private.append(dict(**x,map_name=r['map_name'],episode_id=r['episode_id'],old_category=parsed[i]['category']))
 for name,value in [('public_inputs.json',public),('PRIVATE_sampling.json',private)]: (a.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False))
 manifest=dict(seed=42,N=len(public),groups=dict(collections.Counter(x['group'] for x in private)),source_manifest_SHA=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),bearing_predictions_SHA=hashlib.sha256((a.bearing_run/'confirmation/relations_angle_rows.jsonl').read_bytes()).hexdigest(),public_input_SHA=hashlib.sha256((a.output/'public_inputs.json').read_bytes()).hexdigest(),random_population='Uniform100 of 21578 remaining TRAIN queries after excluding targeted300; not an unbiased full-TRAIN estimate',stratification='Round-robin current automatic relation category; per-query unique; samples are not target-independent',no_GT_in_parser_input=True,no_independent_human_gold=True)
 (a.output/'sampling.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
