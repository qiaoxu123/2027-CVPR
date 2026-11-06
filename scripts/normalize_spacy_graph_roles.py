"""Map native spaCy path roles without counting every noun as an intermediary."""
import argparse,json,shutil,hashlib,time
from pathlib import Path
from scripts.run_parser_benchmark import convert_spacy,infer_type
from scripts.analyze_instruction_landmark_association import PublicNames
from graphnav.parser_benchmark import validate_graph,relation_paths

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();public={r['index']:r for r in json.load(open(a.run/'data/public_inputs.json'))}
 for tag in 'AB':
  folder=a.run/tag;backup=a.run/(tag+'_before_intermediate_role_mapping');assert not backup.exists();shutil.copytree(folder,backup);rows=[json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()];tick=time.perf_counter();changed=0
  for row in rows:
   source=public[row['index']];matcher=PublicNames({'public':{str(i):dict(name=n) for i,n in enumerate(source['landmark_catalog'])}});g=validate_graph(source['text'],convert_spacy(row['raw']['parsed'],matcher.match('public',source['text'])))
   for e in g['entities']:
    if e['type']=='unknown':e['type']=infer_type(e['text']);e['type_source']='shared_lexical_span_heuristic'
   g['chains']=relation_paths(g);changed+=g!=row['graph'];row['graph']=g;row['postprocess_patch']='native_path_internal_nodes_intermediate_other_nouns_reference'
  (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));d=dict(graphs_changed=changed,raw_native_parses_unchanged=True,no_GT_read=True,CPU_s=time.perf_counter()-tick,source_SHA=hashlib.sha256(Path('scripts/run_parser_benchmark.py').read_bytes()).hexdigest());(folder/'postprocess_patch.json').write_text(json.dumps(d,indent=2));print(tag,json.dumps(d))
if __name__=='__main__':main()
