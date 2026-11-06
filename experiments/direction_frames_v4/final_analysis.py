"""Denominator audits, historical context and old-support comparisons."""
from shared import *
from evaluate import metrics
import pandas as pd

def run():
 Q=readlines(OUT/'geometry_checks.jsonl.gz');by={r['occurrence_id']:r for r in Q};P={r['occurrence_id']:r for r in readlines(OUT/'frozen_frames.jsonl.gz')};R=readlines(OUT/'relations.jsonl.gz');X={r['occurrence_id']:r for r in readlines(OUT/'public_inputs.jsonl.gz')};old=readj(V3/'results/strict_case_details.json');comparison=[]
 for o in old:
  r=by[o['occurrence_id']];comparison.append({'occurrence_id':o['occurrence_id'],'old_cohort':o['cohort'],'old_metric':o['metric'],'old_score':o['local_score'],'v4_state':r['state'],'v4_result':r['result'],'v4_metric':r['metric'],'directly_comparable_metric':False,'note':'v3 angular sector/straight side and v4 curved-road field differ; no raw rate subtraction'})
 writej(OUT/'v3_strict_support_audit.json',comparison);pd.DataFrame(comparison).to_csv(OUT/'v3_strict_support_audit.csv',index=False)
 dedup=[]
 for cohort,rr in [('all_unique',[r for r in Q if r['state']=='unique_field']),('new_road_unique',[r for r in Q if r['state']=='unique_field' and r['origin']=='v4_explicit_road_anchor'])]:
  uniq={}
  for r in rr:uniq.setdefault((r['cluster'],r['reference_group'],r['word'],P[r['occurrence_id']]['road_field_key']),r)
  dedup.append({'cohort':cohort,'dedup_unit':'subject/reference/word/frozen-road-field',**metrics(list(uniq.values()))})
 writej(OUT/'deduplicated_sensitivity.json',dedup)
 words=[];reasons=[]
 for w in sorted({r['semantic_word'] for r in R}):
  rr=[r for r in Q if r['word']==w];words.append({'word':w,**metrics(rr)})
  for reason,n in collections.Counter((P[r['occurrence_id']]['reason']) for r in rr).items():reasons.append({'word':w,'frame_or_binding_reason':reason,'n':n})
 writej(OUT/'word_counts.json',words);pd.DataFrame(words).to_csv(OUT/'word_counts.csv',index=False);pd.DataFrame(reasons).to_csv(OUT/'frame_exclusion_reasons.csv',index=False)
 scan=readlines(OUT/'train_semantic_scan.jsonl.gz');lex=[]
 patterns={'A_stand_at_facing':r'\bstand(?:ing)?\s+at\b.*\bfacing\b','B_looking_from_towards':r'\b(?:looking|viewed|viewing)\s+from\b.*\b(?:towards?|at)\b','C_coming_from':r'\bcoming\s+from\b','D_along':r'\balong\b','observer_facing':r'\bfacing\b','approaching':r'\bapproaching\b','viewed_from':r'\bviewed\s+from\b','towards':r'\btowards?\b'}
 for k,pattern in patterns.items():
  rr=[r for r in scan if re.search(pattern,r['instruction'],re.I)];lex.append({'structure':k,'all_TRAIN_instruction_count':len(rr),'with_named_mentions_and_direction_words':sum(bool(r['named_mentions'] and r['direction_words']) for r in rr),'count_unit':'episodes; structures overlap; not independently bound relations'})
 writej(OUT/'semantic_search_coverage.json',lex);pd.DataFrame(lex).to_csv(OUT/'semantic_search_coverage.csv',index=False)
 history={'v1':{'report':'../direction_frames/REPORT_ZH.md','valid_all_direction_pairs':212,'scope':'different conservative target-headed population, not matched to v4; opposite examples later qualified by v2'},'v2_matched_baselines':[r for r in readj(V2/'results/matched_comparisons.json') if r['n']==344],'v3':[r for r in readj(V3/'results/summary.json') if r['scheme']=='object_5fold' and r['cohort'] in {'v2_all','v2_high','expanded_all','roadside_secondary'} and r['model'] in {'strict_rules','semantic_candidates','bert_router'} and r['mode']=='conditional_local'],'historical_files_sha256':{str(p.relative_to(BASE4.parent)):sha(p) for p in [V2/'results/matched_comparisons.json',V3/'results/summary.json',BASE4.parent/'direction_frames/results/manifest.json']},'cross_version_rate_subtraction_valid':False}
 writej(OUT/'historical_context.json',history)
 new=[r for r in Q if r['origin'].startswith('v4')];sc=[r for r in Q if r['state']=='unique_field' and r['result'] in {'consistent','conflict'}];writej(OUT/'evidence_summary.json',{'all_bound':len(X),'full_relations':len(Q),'new_bound':len(new),'unique_scalar_headings':sum(r['state']=='unique' for r in Q),'unique_road_fields':sum(r['state']=='unique_field' for r in Q),'unique_road_fields_scorable':len(sc),'scorable_maps':len({r['map'] for r in sc}),'scorable_reference_groups':len({r['reference_group'] for r in sc}),'scorable_subject_objects':len({r['cluster'] for r in sc}),'scorable_consistent':sum(r['result']=='consistent' for r in sc),'new_relation_instructions':len({r['source_instruction'] for r in new if r['origin']=='v4_explicit_road_anchor'}),'new_road_anchor_relations':sum(r['origin']=='v4_explicit_road_anchor' for r in new)})
 fields=readlines(OUT/'road_fields.jsonl.gz');writej(OUT/'road_field_diagnostics.json',[{k:v for k,v in f.items() if k not in {'points_xy','heading_deg','distance_from_source_m','edges'}} for f in fields])
 print(readj(OUT/'evidence_summary.json'))
if __name__=='__main__':run()
