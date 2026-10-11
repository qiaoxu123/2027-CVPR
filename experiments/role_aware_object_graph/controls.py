"""Matched-alpha and matched-abstention diagnostic controls, never select parameters."""
import argparse,json,time
from pathlib import Path
import torch
from .evaluate import OUT,evaluate,dependency_lookup
from experiments.language_object_graph.run import load,aggregate,dump_rows,METRICS
from experiments.language_object_graph.objects import ObjectGraph
from experiments.bidirectional_core.stats import paired_difference
from experiments.semantic_polygon_grounding.audit import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','dev','seen'],required=True);a=p.parse_args();torch.set_num_threads(1);cfg=json.loads((OUT/'frozen_config.json').read_text())['selection']['A4'];states,labels,_=load(a.split);cache=dependency_lookup(a.split);graphs={p.stem:ObjectGraph.restore(torch.load(p,weights_only=True)) for p in Path('runs/language_object_graph/graphs').glob('*.pt')};result={};rows={}
 for method in ('A3','A4','A4_damped_single'):
  r,_,cost=evaluate(states,labels,graphs,method,cfg['weight'],cfg['backend'],cache);rows[method]=r;result[method]=dict(metrics=aggregate(r),timing=cost);dump_rows(OUT/f'{a.split}_{method}_matched_control.jsonl.gz',r)
 result['differences']={method:{m:paired_difference(rows[method],rows['A4'],m,seed=20261011) for m in METRICS} for method in ('A3','A4_damped_single')};save(OUT/f'{a.split}_matched_controls.json',result)
if __name__=='__main__':main()
