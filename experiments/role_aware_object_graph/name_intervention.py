"""Controlled name-scope intervention: preserve P5 types/factors/alpha exactly.

Clears only names that are outside the new target nominal entity. GT is solely
used afterwards to summarize original-role errors; never decides the edit.
Diagnostic, not model/threshold selection.
"""
import json,time,gzip
from pathlib import Path
import numpy as np
import torch
from .evaluate import OUT
from .parser import parse
from experiments.language_object_graph.language import parse as old_parse
from experiments.language_object_graph.reasoning import infer,mix
from experiments.language_object_graph.run import load,row,aggregate,dump_rows,METRICS
from experiments.language_object_graph.objects import ObjectGraph
from experiments.bidirectional_core.stats import paired_difference
from experiments.semantic_polygon_grounding.audit import save

def main():
 torch.set_num_threads(1);graphs={p.stem:ObjectGraph.restore(torch.load(p,weights_only=True)) for p in Path('runs/language_object_graph/graphs').glob('*.pt')};summary={}
 for split in ('train','dev','seen'):
  states,labels,_=load(split);rows=[];counts={'named_target':0,'wrong_target_named':0,'wrong_name_and_old_goal_error':0,'changed_names':0,'recoveries_in_wrong_names':0,'damage_in_wrong_names':0}
  with gzip.open(f'runs/language_object_graph/{split}_A2_w0.5.jsonl.gz','rt') as f:original=[json.loads(x) for x in f]
  for state,old in zip(states,original):
   m=state['metadata'];g=graphs[m['map_name']];p=state['public'];l=old_parse(m['instruction'],g);new=parse(m['instruction'],g);target=labels[m['sample_id']]['target_object_id'];oldnames=list(l['entities'][0]['named_ids']);keep=set(new['entities'][0]['named_ids']) if new['entities'] else set();wrong=bool(oldnames and g.byid[str(target)] not in oldnames)
   counts['named_target']+=bool(oldnames);counts['wrong_target_named']+=wrong;counts['wrong_name_and_old_goal_error']+=wrong and not old['metrics']['goal_r1']
   cleaned=[i for i in oldnames if i in keep];counts['changed_names']+=cleaned!=oldnames;l['entities'][0]['named_ids']=cleaned;l['entities'][0]['named_names']=[g.names[i] for i in cleaned]
   b=np.asarray(p['bounds']);world=b[:2]+np.asarray(p['grid_xy'])*(b[2:]-b[:2]);result=infer(g,l,world,p['static_logits'],relations=True,yaw=float(p['pose_world'][3]));result['language']=l;logits=mix(p['static_logits'],result['heatmap'],.5);r=row(state,labels[m['sample_id']],g,result,logits)
   delta=r['metrics']['goal_r1']-old['metrics']['goal_r1'];counts['recoveries_in_wrong_names']+=wrong and delta>0;counts['damage_in_wrong_names']+=wrong and delta<0;r['wrong_original_name_role']=wrong;r['name_binding_changed']=cleaned!=oldnames;r['delta_goal_vs_P5']=delta;rows.append(r)
  dump_rows(OUT/f'{split}_name_only_intervention.jsonl.gz',rows);summary[split]=dict(counts=counts,metrics=aggregate(rows),vs_P5={metric:paired_difference(original,rows,metric,seed=20261011) for metric in METRICS},status='Fixed raw-language-only intervention; no GT chooses edits. Counts are direct attribution of name scope, not all role/scope changes.')
 save(OUT/'name_intervention.json',summary)
if __name__=='__main__':main()
