"""Full TRAIN text rescan; no target coordinate or prediction is consulted."""
import argparse,gzip,hashlib,json,re
from collections import Counter
from pathlib import Path
from graphnav.landmark_alias import canonical_name

CARDINAL=re.compile(r'\b(north|south|east|west)\s+(?:side\s+)?of\s+(?:the\s+)?',re.I)
INVERSE={'north':'south','south':'north','east':'west','west':'east'}

def retrieve_candidates(parsed,catalog):
 result=[];scan=[]
 for r in parsed:
  text=r['original_text'];found={};matches=list(CARDINAL.finditer(text))
  for m in matches:
   for lm in r['landmarks']:
    if lm['start'] not in [m.end(),m.end()+4] or text[m.end():lm['start']].strip().lower() not in ['','the']:continue
    for name in lm['public_names']:
     key=(m.group(1).lower(),name);found[key]=dict(word=key[0],name=name,relation_span=[m.start(),lm['end']],reference_span=[lm['start'],lm['end']],retrieval=['lexical_adjacent_public_name'],name_binding='existing_text_only_catalog_span')
   # Unique public-name prefix, e.g. "Full Circle Events building". This
   # uses text/public catalog only and is retained for explicit binding review.
   if not any(v['relation_span'][0]==m.start() for v in found.values()):
    tail=text[m.end():];n=re.match(r'([^,.!?;]+?)\s+building\b',tail,re.I)
    if n:
     alias=canonical_name(n.group(1));names=sorted({x['name'] for x in catalog[r['map_name']].values() if x.get('name') and canonical_name(x['name']).startswith(alias)})
     if len(alias)>=12 and len(names)==1:
      name=names[0];key=(m.group(1).lower(),name);found[key]=dict(word=key[0],name=name,relation_span=[m.start(),m.end()+n.end()],reference_span=[m.end(),m.end()+len(n.group(1))],retrieval=['unique_public_name_prefix'],name_binding='unique_public_prefix_requires_text_review')
  edges={e['id']:e for e in r['relations']}
  for path in r['direct_relations']:
   if path['hops']!=1:continue
   e=edges[path['relations'][0]['relation_id']];predicate=e['predicate'];mm=re.fullmatch(r'(north|south|east|west)(?: side)? of',predicate)
   if not mm:continue
   word=mm[1]
   if path['relations'][0]['traversal']=='reverse':word=INVERSE[word]
   for name in path['landmark_names']:
    key=word,name
    if key in found:found[key]['retrieval'].append('dependency_direct_proposal');continue
    found[key]=dict(word=word,name=name,relation_span=e['source_span'],reference_span=None,retrieval=['dependency_direct_proposal'],name_binding='existing_text_only_catalog_span')
  scan.append(dict(episode=r['episode'],map_name=r['map_name'],cardinal_predicates=len(matches),candidate_count=len(found),text_sha256=hashlib.sha256(text.encode()).hexdigest()))
  for (word,name),item in found.items():
   ids=[str(k) for k,v in catalog[r['map_name']].items() if canonical_name(v.get('name',''))==canonical_name(name)]
   if not ids:continue
   result.append(dict(candidate_id=len(result),episode=r['episode'],map_name=r['map_name'],instruction=text,word=word,landmark_name=name,public_object_ids=ids,public_types=sorted({catalog[r['map_name']][k]['object_type'] for k in ids}),parser_target_category=r['target']['category'],parser_flags=r['ambiguity'],**{k:v for k,v in item.items() if k not in ['word','name']}))
 return result,scan

def main():
 p=argparse.ArgumentParser();p.add_argument('--structured',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 with gzip.open(a.structured,'rt') as f:parsed=json.load(f)
 assert len(parsed)==22002 and len({r['episode'] for r in parsed})==22002
 assert all(r['target_GT_used'] is False for r in parsed)
 cat=json.loads(a.catalog.read_text());candidates,scan=retrieve_candidates(parsed,cat)
 for name,data in [('candidates.json',candidates),('full_scan.json',scan)]: (a.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
 manifest=dict(instructions_scanned=len(scan),lexical_predicates=sum(r['cardinal_predicates'] for r in scan),candidate_pairs=len(candidates),candidate_episodes=len({r['episode'] for r in candidates}),candidate_words=dict(Counter(r['word'] for r in candidates)),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.structured,a.catalog,Path(__file__)]},target_coordinates_used=False,scope='All 22002 original TRAIN texts cached losslessly by prior text-only parser. Candidate retrieval is not verified direct-relation labeling. Compound cardinal/proper-name tokens alone are not predicates.')
 (a.output/'scan_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
