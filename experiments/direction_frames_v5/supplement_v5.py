"""Text-only support audit plus exact F transitions and novelty receipts."""
from v5core import *
from language_v5 import SEARCH
from road_language import extract as frozen_extract

def run():
 new=readlines(RESULT/'new_candidates.jsonl.gz');X={x['occurrence_id']:x for x in readlines(RESULT/'public_inputs.jsonl.gz')};scan=readlines(RESULT/'semantic_scan.jsonl.gz');O=readj(DATA/'cityrefer_original/objects.json');catalogs={m:TypedRoadCatalog([Entity(i,o['name'],o['object_type']) for i,o in obs.items() if o['name']]) for m,obs in O.items() if m in {r['map'] for r in new}};rows=[]
 for r in new:
  emitted=frozen_extract(r['instruction'],catalogs[r['map']]);rows.append({'occurrence_id':r['occurrence_id'],'status':r['status'],'novelty':r['novelty'],'template':r['template'],'frozen_v4_binding_extractor_emits_same_word':any(z['span_start']==r['span_start'] and z['reference_ids']==r['reference_ids'] for z in emitted),'new_road_vs_v4_unique':r['new_road_vs_v4_unique'],'new_map_vs_v4_unique':r['new_map_vs_v4_unique']})
 savecsv('frozen_binding_support',rows)
 raw=[]
 for name,rx in [('coming_from_B_onto_A',r'\bcoming\s+from\b.{0,100}\b(?:onto|into|to)\b'),('turn_onto_A_from_B',r'\bturn(?:\s+(?:left|right))?\s+onto\b.{0,100}\bfrom\b'),('enter_A_via_B',r'\benter(?:ing)?\b.{0,100}\bvia\b'),('approaching_A_from_B',r'\bapproaching\b.{0,100}\bfrom\b'),('along_A_from_B',r'\balong\b.{0,140}\bfrom\b'),('facing_B_from_A',r'\bfacing\b.{0,100}\bfrom\b')]:
  a=[r for r in scan if re.search(rx,r['instruction'],re.I)];template={'coming_from_B_onto_A':'coming_from_to_onto','turn_onto_A_from_B':'turn_onto_from','enter_A_via_B':'enter_via','approaching_A_from_B':'approaching_from_UNDERSPECIFIED','along_A_from_B':'along_from','facing_B_from_A':'facing_from_OBSERVER'}[name];matched=sum(any(z['template']==template for z in r['role_pairs']) for r in scan);raw.append({'requested_structure':name,'raw_TRAIN_instructions':len(a),'instructions_with_any_named_road_roles':sum(bool(r['role_pairs']) for r in a),'named_template_instructions':matched,'count_unit':'raw episodes; patterns overlap; not bound relations'})
 savecsv('requested_structure_inventory',raw)
 methods=readlines(RESULT/'method_predictions.jsonl.gz');a={r['occurrence_id']:r for r in methods if r['method']=='A_frozen_v4_field'};f=[r for r in methods if r['method']=='F_exploratory_smoothed_geometry'];savecsv('F_geometry_transitions',[{'occurrence_id':r['occurrence_id'],'map':r['map'],'A_result':a[r['occurrence_id']]['result'],'F_result':r['result'],'changed':a[r['occurrence_id']]['result']!=r['result'],'GT_used_to_choose_geometry':False,'independent_centerline_ground_truth_available':False} for r in f])
 controls=[]
 for method in sorted({r['method'] for r in methods if r['method'].startswith('negative_')}):
  rr=[r for r in methods if r['method']==method];sc=[r for r in rr if r['result'] in {'consistent','conflict'}];ok=sum(r['result']=='consistent' for r in sc);controls.append({'control':method,'support_n':len(rr),'unique_alternative_fields':sum(r.get('alternative_field_state','unique_field')=='unique_field' for r in rr),'geometry_scorable':len(sc),'confirmed':ok,'confirmed_all_support':ok/len(rr),'conditional_consistency':ok/len(sc) if sc else None,'unknown_and_ambiguous_retained':True})
 savecsv('negative_control_coverage',controls)
 # Record the final state for EVERY original/new candidate, including failures.
 Q=readlines(RESULT/'geometry_checks.jsonl.gz');P={p['occurrence_id']:p for p in readlines(V4/'results/frozen_frames.jsonl.gz')};P.update({p['occurrence_id']:p for p in readlines(RESULT/'new_frames.jsonl.gz')});out=[]
 for q in Q:
  p=P.get(q['occurrence_id'],{});out.append({'occurrence_id':q['occurrence_id'],'subject_ids':p.get('subject_ids',[]),'reference_ids':p.get('reference_ids',[]),'word':q['word'],'frame_type':p.get('frame_type','unknown'),'state':q['state'],'reason':q['reason'],'road_field_key':q['road_field_key'],'language_evidence':p.get('language_evidence',p.get('evidence',[])),'origin_xy':p.get('origin_xy'),'unique_scalar_author_heading_verified':False,'GT_used_for_frame':False,'geometry_result_offline':q['result'],'novelty':q['novelty'],'origin':q['origin']})
 writelines(RESULT/'all_relation_outputs.jsonl.gz',out)
 print('Supplement',len(out),'all outputs;',sum(r['changed'] for r in readj(RESULT/'F_geometry_transitions.json')),'F transitions')
if __name__=='__main__':run()
