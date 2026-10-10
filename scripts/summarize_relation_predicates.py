"""Compute geometric classification diagnostics from the exact exported static candidates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from graphnav.candidate_geometry import FullCandidateDataset
from graphnav.relation_supervision import legal_region
from graphnav.checkpointing import file_sha256, provenance


def main():
    p=argparse.ArgumentParser()
    for key in ('records','predictions-dir','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise ValueError('refusing predicate overwrite')
    dataset=FullCandidateDataset(a.records,max_roads=0)
    source=Path(a.predictions_dir);metrics=json.loads((source/'metrics.json').read_text())
    if metrics['input_sha256']!=file_sha256(a.records):raise ValueError('static source SHA mismatch')
    report=dict(source_sha256=metrics['input_sha256'],provenance=provenance(),models={},
        definition='Post-forward geometric predicate confusion counts using the exact FP32 topk(8) first candidate from exported static predictions. No new classifier or inference; contradictory target predicates are retained. Auxiliary convention thresholds, not navigation Recall.')
    for name,model in metrics['models'].items():
        rows=[json.loads(line)for line in (source/(name+'_predictions.jsonl')).read_text().splitlines()]
        if len(rows)!=len(dataset):raise ValueError('static prediction count mismatch')
        counts=defaultdict(lambda:defaultdict(int))
        for index,prediction in enumerate(rows):
            row=dataset.records[index]
            if prediction['index']!=index or prediction['map_name']!=row['map_name'] or prediction['instruction']!=row['instruction']:raise ValueError('static query alignment mismatch')
            graph=dataset._prepared(index)[0];low=np.array(row['world_bounds'][:2]);span=np.array(row['world_bounds'][2:])-low
            points=low+np.stack([dataset.grid_xy[prediction['top8_cells'][0]],row['target_xy']])*span
            for relation in graph['relations']:
                result=legal_region(points,graph['entities'],relation);c=counts[relation.kind]
                if result is None:c['unreliable_or_unsupported']+=1;continue
                c['relations']+=1;c['predicate_agreement']+=int(result[0]==result[1]);c['target_satisfies']+=int(result[1]);c['prediction_satisfies']+=int(result[0]);c['positive_target_correct']+=int(result[0] and result[1])
                c['TP' if result[0] and result[1] else 'FP' if result[0] else 'FN' if result[1] else 'TN']+=1
        result={}
        for kind,c in counts.items():
            result[kind]=dict(c)
            if c['relations']:result[kind]['predicate_accuracy']=c['predicate_agreement']/c['relations'];result[kind]['target_satisfaction_fraction']=c['target_satisfies']/c['relations']
            if c['target_satisfies']:result[kind]['positive_target_satisfaction_recall']=c['positive_target_correct']/c['target_satisfies']
            result[kind].update(precision=c['TP']/max(c['TP']+c['FP'],1),recall=c['TP']/max(c['TP']+c['FN'],1),F1=2*c['TP']/max(2*c['TP']+c['FP']+c['FN'],1))
        report['models'][name]=dict(checkpoint_sha256=model['checkpoint_sha256'],prediction_sha256=file_sha256(source/(name+'_predictions.jsonl')),by_kind=result)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print('Exact static predictions relation diagnostics complete')


if __name__=='__main__':main()
