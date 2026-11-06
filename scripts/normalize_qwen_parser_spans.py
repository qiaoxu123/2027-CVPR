"""Normalize raw Qwen graphs by unique literal/case-only spans, never GT."""
import argparse,json,time,hashlib,re,shutil
from pathlib import Path
from scripts.run_parser_benchmark import aligned_entity,infer_type
from graphnav.parser_benchmark import validate_graph,relation_paths

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--tag',default='D');a=p.parse_args();folder=a.run/a.tag;assert (folder/'runtime.json').exists(),'Wait for completed inference before rewriting graph normalization';backup=a.run/(a.tag+'_before_case_only_alignment');assert not backup.exists();shutil.copytree(folder,backup);rows=[json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()];assert a.tag!='D' or len(rows)==400,'Final D needs all400 rows';public={r['index']:r for r in json.load(open(a.run/'data/public_inputs.json'))};tick=time.perf_counter();changed=0;case_repairs=0
 for row in rows:
  if not isinstance(row['raw'],str):continue
  try:
   raw=re.sub(r'^```(?:json)?\s*','',row['raw'].strip());raw=re.sub(r'\s*```$','',raw);obj=json.loads(raw)
  except Exception:continue
  source=public[row['index']]['text'];es=[];notes=[]
  for j,e in enumerate(obj.get('entities',[])):
   entity,note=aligned_entity(source,e,j)
   if entity:es.append(entity)
   if note:notes.append(note);case_repairs+=note=='span_aligned_from_unique_case_only_literal'
  obj['entities']=es;g=validate_graph(source,obj)
  for e in g['entities']:
   if e['type']=='unknown':e['type']=infer_type(e['text']);e['type_source']='shared_lexical_span_heuristic'
  g['chains']=relation_paths(g);changed+=g!=row['graph'];row['graph']=g;row['alignment_warnings']=notes;row['postprocess_patch']='unique_case_only_source_span_alignment_no_semantic_rewrite'
 (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));result=dict(rows=len(rows),graphs_changed=changed,case_only_entity_alignments=case_repairs,CPU_s=time.perf_counter()-tick,model_generations_unchanged=True,no_GT_read=True,source_SHA=hashlib.sha256(Path('scripts/run_parser_benchmark.py').read_bytes()).hexdigest());(folder/'postprocess_patch.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
