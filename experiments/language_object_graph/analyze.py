"""Read-only paired analysis, baseline parity and illustrative fixed-score export."""
import csv,gzip,json,platform,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from .run import OUT,load
from .objects import ObjectGraph
from .pipeline import forward
from .reasoning import probability
from experiments.bidirectional_core.stats import paired_difference,summarize_rows
from experiments.semantic_polygon_grounding.audit import save,sha


def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]

def main():
    main=json.loads((OUT/'metrics.json').read_text());graphs={};analysis={};csvrows=[]
    for split in ('train','dev','seen'):
        states,labels,_=load(split);methods={}
        for method in ('A0','A1','A2','IdentityOracle'):
            weight=0 if method=='A0' else .5 if method=='IdentityOracle' else main['selected'][method];key=f'{method}_w{weight:g}'
            rows=read(OUT/f'{split}_{key}.jsonl.gz');methods[method]=rows
            if method!='IdentityOracle':
                # Same frozen public scores, independently verify stored rankings;
                # no metrics/config search, solely export missing heatmap scores.
                with gzip.open(OUT/f'{split}_{method}_goal_scores.jsonl.gz','wt') as out:
                    for s,r in zip(states,rows):
                        assert s['metadata']['sample_id']==r['sample_id'];name=r['map_name']
                        if name not in graphs:graphs[name]=ObjectGraph.restore(torch.load(OUT/'graphs'/f'{name}.pt',weights_only=True))
                        result=forward(graphs[name],s['metadata']['instruction'],s['public'],method=method,weight=weight)
                        logits=np.asarray(s['public']['static_logits']) if method=='A0' else result['logits']
                        assert np.argsort(logits,kind='stable')[::-1].tolist()==r['goal_order']
                        out.write(json.dumps(dict(sample_id=r['sample_id'],goal_logits=logits.tolist(),goal_probabilities=probability(logits).tolist()))+'\n')
            m=main['metrics'][split][key]
            csvrows.append(dict(split=split,method=method,families=len(rows),maps=m['goal_r1']['maps'],r1=100*m['goal_r1']['mean'],r4=100*m['goal_r4']['mean'],r8=100*m['goal_r8']['mean'],ne=m['ne']['mean'],recovery=round(m['recovery']['mean']*len(rows)),damage=round(m['damage']['mean']*len(rows)),object_r1=None if method=='A0' else 100*m['object_r1']['mean'],object_r8=None if method=='A0' else 100*m['object_r8']['mean']))
        factors=Counter();coverage=Counter();by_map=defaultdict(list);objects=set()
        for s,r in zip(states,methods['A2']):
            assert s['metadata']['sample_id']==r['sample_id'];l=r['language_graph'];coverage['families']+=1;coverage['target_type_parsed']+=l['entities'][0]['object_type'] is not None;coverage['named_target']+=bool(l['entities'][0]['named_ids']);coverage['relation_trigger']+=bool(l['factors']);coverage['effective_relation']+=bool(r['relation_hypotheses']);coverage['multiple_references']+=len(l['entities'])>=3
            coverage['ambiguous_relative_frame']+=any(len(f['frame_hypotheses'])>1 for f in l['factors']);coverage['unknown_reference_entity']+=any(not e['named_ids'] and e['object_type'] is None for e in l['entities'][1:])
            for factor in l['factors']:factors[factor['kind']]+=1
            by_map[r['map_name']].append(r);objects.add((r['map_name'],str(labels[r['sample_id']]['target_object_id'])))
        relation_delta={k:paired_difference(methods['A1'],methods['A2'],k,seed=20261011) for k in ('goal_r1','goal_r4','goal_r8','ne','object_r1','object_r4','object_r8')}
        analysis[split]=dict(coverage=dict(coverage),factors=dict(factors),independent_objects=len(objects),relation_delta_vs_A1=relation_delta,
            by_map={name:{k:summarize_rows(rows,k,seed=20261011) for k in ('goal_r1','recovery','damage','ne')} for name,rows in by_map.items()},
            A0_object_metrics_status='A0 has no native Object Belief. Raw evaluator object_* is an A1 unary shadow probe only; omitted from main comparison.csv and report baseline.')
        if split=='seen':
            legacy={r['sample_id']:r for r in map(json.loads,Path('runs/instance_grounding/evaluation_seen/geometry_only_predictions.jsonl').read_text().splitlines())}
            for r in methods['A0']:
                for key in ('goal_r1','goal_r4','goal_r8','ne'):assert abs(r['metrics'][key]-legacy[r['sample_id']]['metrics'][key])<1e-6,(r['sample_id'],key)
            analysis[split]['A0_parity']='128/128 primary families match existing frozen A3 metrics and tie order'
    save(OUT/'analysis.json',analysis)
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csvrows)
    save(OUT/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),cuda=torch.version.cuda,training='No CityNav head/backbone training: Dev gate failed',cache_semantics='A0 cached frozen A3, full public object geometry, no new RGB encoding in main ablations'))

if __name__=='__main__':main()
