"""All-TRAIN public language parsing and precommitted map split. No GT parsing."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import spacy
from graphnav.role_relation_language import parse_doc
from scripts.analyze_instruction_landmark_association import PublicNames
from graphnav.checkpointing import file_sha256


def main():
 p=argparse.ArgumentParser()
 for k in ['data-manifest','public-objects','old-protocol','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 data=json.loads(a.data_manifest.read_text());records=data['records'];public=json.loads(a.public_objects.read_text());old=json.loads(a.old_protocol.read_text());dev=old['split']['heldout_maps'];maps=sorted({r['map_name'] for r in records});confirm=[]
 for city in ['birmingham','cambridge']:
  choices=[m for m in maps if m.startswith(city) and m not in dev]
  confirm+=sorted(choices,key=lambda m:hashlib.sha256(('relation-confirm-42'+m).encode()).hexdigest())[:2]
 groups={k:[i for i,r in enumerate(records) if (r['map_name'] in confirm if k=='confirm' else r['map_name'] in dev if k=='dev' else r['map_name'] not in confirm+dev)] for k in ['fit','dev','confirm']}
 # Public target identity is used exclusively to isolate duplicated objects,
 # never to parse text, retrieve neighbors or create network inputs.
 def identity(r):
  parts=str(r['episode_id']).split(':');obj=public[r['map_name']].get(parts[1],{});xy=obj.get('contour',[])
  if xy:return hashlib.sha256(json.dumps(sorted(tuple(round(float(v),5) for v in p) for p in xy)).encode()).hexdigest()
  return ':'.join(parts[:2])
 seen_text=set();seen_identity=set();purged=[]
 for k in ['confirm','dev','fit']:
  keep=[]
  for i in groups[k]:
   row=records[i];text=' '.join(row['instruction'].casefold().split());oid=identity(row)
   if text in seen_text or oid in seen_identity:purged.append(dict(index=i,split=k));continue
   keep.append(i)
  groups[k]=keep;seen_text.update(' '.join(records[i]['instruction'].casefold().split()) for i in keep);seen_identity.update(identity(records[i]) for i in keep)
 split=dict(indices=groups,maps={k:sorted({records[i]['map_name'] for i in v}) for k,v in groups.items()},counts={k:len(v) for k,v in groups.items()},purged=purged,GT_used_for_parse=False,GT_identity_only_for_split=True,source_data_SHA=file_sha256(a.data_manifest),catalog_SHA=file_sha256(a.public_objects))
 (a.output/'split.json').write_text(json.dumps(split,indent=2));print(json.dumps({k:v for k,v in split.items() if k!='indices'},indent=2),flush=True)
 nlp=spacy.load('en_core_web_sm',disable=['ner']);matcher=PublicNames(public);out=[];counter=Counter()
 for i,(row,doc) in enumerate(zip(records,nlp.pipe([r['instruction'] for r in records],batch_size=128))):
  parsed=parse_doc(doc,row['instruction'],row['map_name'],matcher);out.append(dict(index=i,**parsed));counter[parsed['category']]+=1
  if (i+1)%1000==0:print(json.dumps(dict(parsed=i+1,counts=dict(counter))),flush=True)
 with (a.output/'parsed.jsonl').open('w') as f:
  for q in out:f.write(json.dumps(q)+'\n')
 available=set(groups['fit']+groups['dev']);sample=[]
 for category in ['direct_geometry','indirect_geometry','uncertain','no_confirmed_anchor']:
  ids=[i for i in available if out[i]['category']==category];ids.sort(key=lambda i:hashlib.sha256(('audit42'+str(i)).encode()).hexdigest());sample += [out[i] for i in ids[:10]]
 (a.output/'parse_audit_sample.json').write_text(json.dumps(sample,indent=2))
 (a.output/'parser_summary.json').write_text(json.dumps(dict(N=len(out),counts=dict(counter),spacy=spacy.__version__,model=nlp.meta['version'],GT_category_or_ID_at_parser=False),indent=2));print('Parsing complete',flush=True)
if __name__=='__main__':main()
