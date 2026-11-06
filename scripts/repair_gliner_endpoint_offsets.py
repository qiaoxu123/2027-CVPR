"""Re-normalize saved GLiNER outputs using their actual spans, without inference."""
import argparse,json,hashlib,time,shutil
from pathlib import Path
from graphnav.parser_benchmark import validate_graph,relation_paths
from scripts.run_parser_benchmark import gliner_graph,infer_type

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();folder=a.run/'C';backup=a.run/'C_before_endpoint_fix';assert not backup.exists();shutil.copytree(folder,backup);data={r['index']:r for r in json.load(open(a.run/'data/public_inputs.json'))};rows=[json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()];tick=time.perf_counter();changed=0
 for row in rows:
  src=data[row['index']];old=row['graph'];g=validate_graph(src['text'],gliner_graph(row['raw'],src['text'],src['landmark_catalog']))
  for e in g['entities']:
   if e['type']=='unknown':e['type']=infer_type(e['text']);e['type_source']='shared_lexical_span_heuristic'
  g['chains']=relation_paths(g);changed+=g!=old;row['graph']=g;row['postprocess_patch']='explicit_GL iNER_model_endpoint_offsets_v2'.replace('GL iNER','GLiNER')
 (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));result=dict(rows=len(rows),graphs_changed=changed,cpu_postprocess_s=time.perf_counter()-tick,no_new_inference=True,no_GT_read=True,raw_outputs_preserved=True,postprocess_source_SHA=hashlib.sha256(Path('scripts/run_parser_benchmark.py').read_bytes()).hexdigest());(folder/'postprocess_patch.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
