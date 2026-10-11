"""Isolated frozen spaCy CLI: public raw text only, labels never loaded."""
import argparse,json,time,hashlib
from pathlib import Path
import spacy

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();raw=json.loads(Path(a.input).read_text());nlp=spacy.load('en_core_web_sm',exclude=['ner','lemmatizer']);start=time.perf_counter();rows={}
 for item,doc in zip(raw,nlp.pipe([r['instruction'] for r in raw],batch_size=32)):
  text=item['instruction'];rows[hashlib.sha256(text.encode()).hexdigest()]=dict(instruction=text,tokens=[dict(text=t.text,start=t.idx,end=t.idx+len(t.text),pos=t.pos_,dep=t.dep_,head=t.head.i) for t in doc],noun_chunks=[dict(start=x.start_char,end=x.end_char,head=x.root.i) for x in doc.noun_chunks])
 Path(a.output).write_text(json.dumps(dict(model='en_core_web_sm',model_version=nlp.meta['version'],spacy_version=spacy.__version__,seconds=time.perf_counter()-start,documents=len(raw),parses=rows),ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
