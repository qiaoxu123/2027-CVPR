"""Final comparisons on identical support, uncertainty and concise coverage outputs."""
from common import *
import pandas as pd

def run(out):
 R=readlines(out/'evaluation_predictions.jsonl.gz');groups=collections.defaultdict(dict)
 for r in R:groups[r['scheme'],r['cohort'],r['model']][r['occurrence_id']]=r
 dif=[]
 for scheme in ['object_5fold','leave_map_out']:
  for cohort in ['v2_all','v2_high','expanded_all','expanded_high','expanded_excluding_known_boat','roadside_secondary']:
   for a,b,label in [('bert_router','bert_support_uniform_orientation','BERT_geometry_beyond_uniform_same_support'),('semantic_candidates','semantic_no_road_facing_prior','mentioned_road_facing_prior_ablation'),('bert_router','semantic_candidates','BERT_versus_simple_semantic_rules')]:
    x=groups[scheme,cohort,a];y=groups[scheme,cohort,b];assert set(x)==set(y);rr=list(x.values());delta=[r['hybrid_score']-y[r['occurrence_id']]['hybrid_score'] for r in rr]
    if a=='bert_router' and b=='bert_support_uniform_orientation':assert all((x[k]['local_score'] is None)==(y[k]['local_score'] is None) for k in x)
    dif.append({'scheme':scheme,'cohort':cohort,'comparison':label,'n':len(rr),'difference':float(np.mean(delta)),'cluster95':cluster_ci(rr,delta),'reference_cluster95':cluster_ci([{**r,'cluster':r['reference_group']} for r in rr],delta),'map_cluster95':cluster_ci([{**r,'cluster':r['map']} for r in rr],delta),'same_evaluation_population':True})
 writej(out/'ablation_differences.json',dif);pd.DataFrame(dif).to_csv(out/'ablation_differences.csv',index=False)
 X=readlines(out/'public_inputs.jsonl.gz');P=readlines(out/'inference_predictions.jsonl.gz');G=readlines(out/'evaluation_gt.jsonl.gz');byG={r['occurrence_id']:r for r in G};rows=[]
 for x,p in zip(X,P):
  for c in p['candidates']:
   rows.append({'occurrence_id':x['occurrence_id'],'map':x['map'],'word':x['word'],'candidate_type':c['type'],'family':c['family'],'level':c['level'],'heading_deg':c['heading_deg'],'strict_selected':c['candidate_id'] in p['strict_ids'],'geometry_status':byG[x['occurrence_id']]['geometry_status'],'source_ids':c['source_ids'],'confidence_rule_score':c['confidence_rule_score'],'evidence':c['evidence']})
 pd.DataFrame(rows).to_csv(out/'all_frame_candidates.csv',index=False)
 cov=[]
 for kind in sorted({c['candidate_type'] for c in rows}):
  q=[r for r in rows if r['candidate_type']==kind and r['word'] in MAIN];ids={r['occurrence_id'] for r in q};strict={r['occurrence_id'] for r in q if r['strict_selected']};eval_ids={rid for rid in ids if byG[rid]['geometry_status']=='disjoint'}
  cov.append({'type':kind,'core_relations_with_candidate':len(ids),'single_explicit_selected':len(strict),'candidate_on_disjoint_core':len(eval_ids),'unique_maps':len({r['map'] for r in q}),'whole_core_population':readj(out/'frame_coverage.json')['core_landmark_candidates'],'no_GT_selected_frame':True})
 writej(out/'candidate_type_coverage.json',cov);pd.DataFrame(cov).to_csv(out/'candidate_type_coverage.csv',index=False)
 strict=[r for r in R if r['scheme']=='object_5fold' and r['model']=='strict_rules' and r['cohort'] in {'expanded_all','roadside_secondary'} and r['local_score'] is not None]
 writej(out/'strict_case_details.json',[{**r,'instruction':next(x['instruction'] for x in X if x['occurrence_id']==r['occurrence_id'])} for r in strict])
 print('ablation and candidate coverage complete',flush=True)
if __name__=='__main__':run(BASE/'results')
