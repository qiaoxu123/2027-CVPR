"""Descriptive parser comparison; true metrics require independent human graphs."""
import argparse,json,csv,collections,shutil,hashlib,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.parser_benchmark import score_reviewed,graph_sets
from scripts.analyze_instruction_landmark_association import PublicNames

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gold',type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=True);inputs=json.load(open(a.run/'data/public_inputs.json'));private={x['index']:x for x in json.load(open(a.run/'data/PRIVATE_sampling.json'))};reviews=json.load(open(a.gold)) if a.gold else [];models={};summary={};descriptive=[]
 for tag in 'ABCD':
  path=a.run/tag/'predictions.jsonl'
  if not path.exists():
   if a.partial:continue
   raise RuntimeError('Missing parser '+tag)
  rows=[json.loads(s) for s in path.read_text().splitlines()];assert len({x['index'] for x in rows})==len(rows)
  if not a.partial:assert len(rows)==len(inputs)==400
  models[tag]={x['index']:x for x in rows};(o/tag).mkdir(exist_ok=True)
  for file in ['predictions.jsonl','runtime.json','config.json','SOURCE.py','postprocess_patch.json']:
   if (a.run/tag/file).exists():shutil.copyfile(a.run/tag/file,o/tag/file)
  groups=collections.defaultdict(list)
  for x in rows:
   es=x['graph']['entities'];edges=x['graph']['relations'];chains=x['graph']['chains'];source=next(r for r in inputs if r['index']==x['index']);catalog={str(j):dict(name=n) for j,n in enumerate(source['landmark_catalog'])};matches=PublicNames({'public':catalog}).match('public',source['text']);expected={(q['start'],q['end']) for q in matches};predicted={(e['start'],e['end']) for e in es if e['role']=='landmark'}
   raw_target=False
   if tag in ['A','B']:raw_target=bool(x.get('raw',{}).get('parsed',{}).get('target_span'))
   elif tag=='C':raw_target=bool(x.get('raw',{}).get('entities',{}).get('navigation target'))
   else:
    try:
     raw_clean=re.sub(r'^```(?:json)?\s*','',x['raw'].strip());raw_clean=re.sub(r'\s*```$','',raw_clean);raw_target=any(e.get('role')=='target' for e in json.loads(raw_clean).get('entities',[]))
    except (ValueError,TypeError,AttributeError):pass
   trusted_paths=sum(z.get('semantic_confident',False) for z in x.get('raw',{}).get('parsed',{}).get('paths',[])) if isinstance(x.get('raw'),dict) else None
   d=dict(raw_target_output=raw_target,effective_trusted_paths_existing_spacy_rules=trusted_paths,index=x['index'],parser=tag,instruction=source['text'],target_description=x['graph'].get('target_description',''),entities_json=json.dumps(es,ensure_ascii=False),relations_json=json.dumps(edges,ensure_ascii=False),chains_json=json.dumps(chains,ensure_ascii=False),group=private[x['index']]['group'],runtime_error=bool(x['error']),validation_errors=len(x['graph']['validation_errors']),unalignable_entities=sum(str(w).startswith(('nonunique/nonliteral','empty entity')) for w in x.get('alignment_warnings',[])),targets=sum(e['role']=='target' for e in es),landmarks=sum(e['role']=='landmark' for e in es),intermediates=sum(e['role']=='intermediate' for e in es),relations=len(edges),has_direct=any(q['hops']==1 for q in chains),has_indirect=any(q['hops']>1 for q in chains),lexical_named_expected=len(expected),lexical_named_predicted=len(predicted),lexical_named_intersection=len(expected&predicted),latency_s=x['amortized_latency_s']);groups[d['group']].append(d);groups['ALL'].append(d);descriptive.append(d)
  stats={}
  for key,v in groups.items():
   stats[key]=dict(N=len(v),runtime_errors=sum(x['runtime_error'] for x in v),entity_span_errors=sum(x['validation_errors'] for x in v),unalignable_entity_mentions=sum(x['unalignable_entities'] for x in v),raw_target_output_coverage=float(np.mean([x['raw_target_output'] for x in v])),target_output_coverage=float(np.mean([x['targets']>0 for x in v])),multiple_targets=sum(x['targets']>1 for x in v),intermediate_output_coverage=float(np.mean([x['intermediates']>0 for x in v])),direct_output_rate=float(np.mean([x['has_direct'] for x in v])),indirect_output_rate=float(np.mean([x['has_indirect'] for x in v])),mean_relations=float(np.mean([x['relations'] for x in v])),catalog_literal_span_overlap=sum(x['lexical_named_intersection'] for x in v),catalog_literal_span_expected=sum(x['lexical_named_expected'] for x in v),catalog_literal_span_predicted=sum(x['lexical_named_predicted'] for x in v))
  review_groups={key:score_reviewed(rows,[review for review in reviews if review['index'] in {x['index'] for x in values}]) for key,values in groups.items()}
  summary[tag]=dict(descriptive=stats,independent_human_metrics=score_reviewed(rows,reviews),independent_human_metrics_by_group=review_groups,runtime=json.load(open(a.run/tag/'runtime.json')) if (a.run/tag/'runtime.json').exists() else None)
 agreements={}
 for tag in models:
  if tag=='A':continue
  common=set(models['A'])&set(models[tag]);entity_j=[];triple_j=[]
  for i in common:
   ea,ta=graph_sets(models['A'][i]['graph']);eb,tb=graph_sets(models[tag][i]['graph']);entity_j.append(len(ea&eb)/max(1,len(ea|eb)));triple_j.append(len(ta&tb)/max(1,len(ta|tb)))
  agreements['A_'+tag]=dict(N=len(common),entity_Jaccard=float(np.mean(entity_j)),directed_triple_Jaccard=float(np.mean(triple_j)),label='Agreement only, neither output is gold; low overlap can be improvement or error')
 (o/'summary.json').write_text(json.dumps(dict(parsers=summary,agreements=agreements,independent_human_gold_available=any(v['independent_human_metrics']['reviewed_count'] for v in summary.values()),accuracy_claim_allowed=any(v['independent_human_metrics']['reviewed_count'] for v in summary.values())),indent=2))
 with (o/'per_instruction.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(descriptive[0]),lineterminator='\n');w.writeheader();w.writerows(descriptive)
 shutil.copyfile(a.run/'data/public_inputs.json',o/'public_inputs.json');shutil.copyfile(a.run/'data/PRIVATE_sampling.json',o/'sampling_outcomes_POSTHOC.json');shutil.copyfile(a.run/'data/sampling.json',o/'sampling.json')
 fig,axes=plt.subplots(1,3,figsize=(13,4));tags=list(summary)
 for ax,key,title in zip(axes,['target_output_coverage','intermediate_output_coverage','indirect_output_rate'],['Target output coverage (NOT accuracy)','Intermediate output coverage (NOT recall)','Indirect chain output rate (NOT accuracy)']):
  ax.bar(tags,[summary[k]['descriptive']['ALL'][key]*100 for k in tags]);ax.set(title=title,ylabel='% of400 instructions');ax.set_ylim(0,100)
 fig.tight_layout();fig.savefig(o/'output_coverage.png',dpi=150);plt.close(fig)
 print(json.dumps({k:v['descriptive']['ALL'] for k,v in summary.items()},indent=2))
if __name__=='__main__':main()
