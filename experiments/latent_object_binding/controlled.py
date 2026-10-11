"""Text-only reviewed structure intervention, never selected using target XY.

These seven explicit clauses are researcher's text review, not independent
expert annotation nor a globally valid variable-parse oracle. Absence/paint
markings and fine shapes not handled by the matcher remain unsupported.
"""
import dataclasses,json, numpy as np,torch
from pathlib import Path
from experiments.language_object_graph.language import entity
from experiments.semantic_polygon_grounding.audit import save
from .resources import OUT,baseline,public
from .language import parse
from .inference import prepare,posterior
from .evaluate import result_row,metrics,aggregate,dump
from experiments.bidirectional_core.stats import paired_difference


def reviewed(g,instruction,annotation,confidence):
    entities=[]
    for i,phrase in enumerate([annotation['target']]+annotation['anchors']):
        e=dataclasses.asdict(entity(phrase,'target' if i==0 else 'reference',g))
        if e['named_ids']:e['object_type']=g.types[e['named_ids'][0]]
        e.update(variable='Z' if i==0 else f'X{i}',head_text=e['object_type'] or 'object',coreference=None);entities.append(e)
    kind=annotation['relation'];factors=[]
    if kind:factors=[dict(kind=kind,subject=0,anchors=list(range(1,len(entities))),subject_hypotheses=[dict(subject=0,probability=1.)],frame_unknown=False,symmetric_anchors=kind=='between')]
    return dict(instruction=instruction,entities=entities,variables=entities,factors=factors,confidence=confidence,hypotheses=[dict(target=0,probability=1.)],source='seven explicit raw clauses, text-only researcher review, NOT independent expert truth',unknown=['fine paint/absence attributes unsupported'],reference_instance_truth=None)


def main():
    torch.set_num_threads(1);a3,source,graphs,proof=baseline();annotations={a['instruction']:a for a in json.loads(Path(__file__).with_name('controlled_text.json').read_text())};maps=torch.load(OUT/'full_geometry/maps.pt',weights_only=True);labels={l['query_index']:l for l in torch.load(OUT/'full_geometry/labels.pt',weights_only=True)};rows={k:[] for k in ('Current','ReviewedSameConfidence','ReviewedOracle')};structures=[]
    with torch.no_grad():
        for q in source['queries']:
            if q['split'] not in ('train','dev') or q['instruction'] not in annotations:continue
            g=graphs[q['map_name']];p=public(q,a3);lang=parse(q['instruction'],g);versions={'Current':lang,'ReviewedSameConfidence':reviewed(g,q['instruction'],annotations[q['instruction']],lang['confidence']),'ReviewedOracle':reviewed(g,q['instruction'],annotations[q['instruction']],1.)}
            # No label access before all three inference outputs exist.
            predictions={k:posterior(prepare(g,q['instruction'],p,language=v)) for k,v in versions.items()}
            label=labels[q['query_index']];bm=metrics(a3['logits'][q['query_index']].numpy(),predictions['Current']['object_belief'].numpy(),maps[q['map_name']],label)[0]
            for k,pred in predictions.items():rows[k].append(result_row(q,label,pred,maps,a3['logits'][q['query_index']],bm['goal_r1']))
            structures.append(dict(query_index=q['query_index'],map_name=q['map_name'],split=q['split'],current=lang,reviewed=versions['ReviewedSameConfidence']))
    for k,v in rows.items():dump(OUT/f'controlled_text_{k}.jsonl.gz',v)
    save(OUT/'controlled_text_metrics.json',dict(n=len(rows['Current']),unique_instructions=len({r['instruction'] for r in rows['Current']}),metrics={k:aggregate(v) for k,v in rows.items()},comparisons={k:{m:paired_difference(rows['Current'],v,m,seed=20261011) for m in ('object_r1','goal_r1','ne')} for k,v in rows.items() if k!='Current'},structures=structures,annotation='Researcher text-only semantic review of seven explicit real instructions, not independent full corpus truth. No GT identity/coordinates used to construct or select structures.',global_parser_oracle_n=0,attribute_oracle_n=0))
if __name__=='__main__':main()
