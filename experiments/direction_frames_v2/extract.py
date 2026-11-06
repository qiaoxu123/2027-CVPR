"""Produce immutable text-only bindings before loading coordinates for diagnosis."""
import argparse,collections,gzip,hashlib,json,re
from pathlib import Path
import spacy
from text_relations import Entity,TextCatalog,parse_landmark_relations,TOKEN,PATTERNS
BASE=Path(__file__).parent

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def norm_text(s):return ' '.join(s.lower().split())
def run(root,out):
 out.mkdir(parents=True,exist_ok=True)
 paths=[root/'original_citynav/citynav_train_seen.json',root/'cityrefer_original/objects.json']
 episodes=json.loads(paths[0].read_text());objects=json.loads(paths[1].read_text())
 maps=sorted({f"{e['area']}_block_{e['block']}" for e in episodes})
 catalogs={m:TextCatalog([Entity(str(i),o['name'],o['object_type']) for i,o in objects[m].items() if o['name']]) for m in maps}
 # Only names/types survive to the NLP phase. Model inference sees only original text.
 types={(m,str(i)):o['object_type'] for m in maps for i,o in objects[m].items()}
 annotation_texts={(m,str(i)):o['descriptions'] for m in maps for i,o in objects[m].items()}
 del objects
 def relevant(e):
  text=e['descriptions'][0];m=f"{e['area']}_block_{e['block']}";hits=catalogs[m].spans(text)
  for match in TOKEN.finditer(text):
   if any(h['start']<=match.start()<h['end'] for h in hits):continue
   rel=PATTERNS[match[0].lower()].match(text[match.start():])
   if rel:
    after=match.start()+rel.end()
    if any(h['start']>=after and re.fullmatch(r'\s*(?:the\s+)?',text[after:h['start']],re.I) for h in hits):return True
  return bool(hits and re.search(r'\bstand(?:ing)?\s+at\b',text,re.I) and re.search(r'\bfacing\b',text,re.I) and re.search(r'\byour\s+(?:right|left|front|behind)\b',text,re.I))
 texts=sorted({e['descriptions'][0] for e in episodes if relevant(e)})
 nlp=spacy.load('en_core_web_sm',disable=['ner']);docs={}
 with gzip.open(out/'dependency_tokens.jsonl.gz','wt',encoding='utf8') as fp:
  for text,doc in zip(texts,nlp.pipe(texts,batch_size=128)):
   docs[text]=doc
   fp.write(json.dumps({'text':text,'tokens':[[t.i,t.idx,t.text,t.lemma_,t.pos_,t.dep_,t.head.i] for t in doc]},ensure_ascii=False)+'\n')
 old=json.loads((BASE.parent/'direction_frames/results/occurrences.json').read_text());oldidx={r['occurrence_id']:r for r in old};rows=[];out_of_scope=0
 for i,e in enumerate(episodes):
  text=e['descriptions'][0]
  if text not in docs:
   out_of_scope+=len(TOKEN.findall(text));continue
  m=f"{e['area']}_block_{e['block']}";tid=str(e['object_ids'][0]);ann=e['ann_ids'][0];desc=annotation_texts[m,tid]
  aligned='exact' if ann<len(desc) and text==desc[ann] else 'whitespace_case_only' if ann<len(desc) and norm_text(text)==norm_text(desc[ann]) else 'unmatched'
  relations,skipped=parse_landmark_relations(docs[text],types[m,tid],catalogs[m]);out_of_scope+=skipped
  for r in relations:
   rid=f"{m}:{tid}:{ann}:{r['span_start']}"
   r.update(occurrence_id=rid,episode=i,map=m,target_id=tid,target_category=types[m,tid],ann_id=ann,instruction=text,annotation_alignment=aligned,cluster=f'{m}:{tid}')
   if r['status']=='bound' and aligned=='unmatched':r.update(status='excluded',reason='annotation_text_mismatch',confidence_tier='excluded',parse_confidence=0.)
   if r['subject_role']=='annotated_target':r['subject_ids']=[tid]
   if r['status']=='bound' and len(r['subject_ids'])!=1:r.update(status='excluded',reason='composite_subject_not_independently_resolved',confidence_tier='excluded',parse_confidence=0.)
   if r['status']=='bound' and set(r['subject_ids'])&set(r['reference_ids']):r.update(status='excluded',reason='target_in_reference_entity',confidence_tier='excluded',parse_confidence=0.)
   prev=oldidx[rid];r.update(v1_status=prev['status'],v1_reason=prev['reason'])
   rows.append(r)
 assert len(rows)+out_of_scope==len(old)==28077 and len({r['occurrence_id'] for r in rows})==len(rows)
 with gzip.open(out/'text_relations.jsonl.gz','wt',encoding='utf8') as fp:
  for r in rows:fp.write(json.dumps(r,ensure_ascii=False)+'\n')
 import pandas as pd
 pd.DataFrame(rows).to_csv(out/'text_relations.csv',index=False)
 manifest={'base_commit':'e6ddb0986b03ac5402894e0f71ffb1f388e513d2','input_files':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in paths],'episodes':len(episodes),'train_maps':maps,'landmark_relation_occurrences':len(rows),'out_of_scope_word_occurrences_not_analyzed':out_of_scope,'population':'direction relation whose actual reference NP is a named landmark; relation subject need not be navigation target','unique_landmark_related_instruction_texts':len(texts),'parser':{'spacy_version':spacy.__version__,'model':'en_core_web_sm','model_version':nlp.meta['version'],'weights_trained_here':False,'model_inputs':'instruction text only'},'text_binding_sha256':hashlib.sha256((out/'text_relations.jsonl.gz').read_bytes()).hexdigest(),'confidence':'rule scores, not calibrated probabilities; high>=.95; medium>=.8; no GT/frame fit in score','same_name_roads':'preserve all segments as a composite entity; medium confidence; never choose segment by target GT','geography':'no new independent geographic calibration; XY/image affine only'}
 dump(out/'manifest.json',manifest)
 print('extraction',dict(collections.Counter((r['status'],r['confidence_tier']) for r in rows)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True,type=Path);ap.add_argument('--out',type=Path,default=BASE/'results');a=ap.parse_args();run(a.data_root,a.out)
