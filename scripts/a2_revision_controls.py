"""CPU frozen128-target TRAIN checks; uncalibrated intervention diagnostics only."""
import hashlib,json
from collections import defaultdict
import torch
from experiments.a2_waldo_landmark_belief.paths import ROOT,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.revision_data import RevisionViews
from experiments.a2_waldo_landmark_belief.revision_model import RevisionNetwork
from experiments.a2_waldo_landmark_belief.revision_runtime import apply_revision
from experiments.a2_waldo_landmark_belief.revision_controls import CONTROLS,controlled_inputs


def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    checkpoint=BASE/'revision_v1/training/best.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=RevisionNetwork(**state['model_config']);model.load_state_dict(state['model']);model.eval()
    dataset=RevisionViews(BASE/'collected_snapshot','train');groups=defaultdict(list)
    for i,entry in enumerate(dataset.entries):
        source=dataset.source[entry['episode']];groups[(source['map_name'],source['object_id'])].append(i)
    chosen=sorted(groups,key=lambda g:hashlib.sha256(f'revision-controls:0:{g}'.encode()).digest())[:128]
    selected=[min(groups[g],key=lambda i:hashlib.sha256(dataset.entries[i]['filename'].encode()).digest()) for g in chosen]
    totals={c:dict(views=0,correct=0,ne_sum=0.,updated=0,changed_peak=0,exact_A0=0) for c in ('A0',*CONTROLS)};records=[]
    with torch.inference_mode():
        for i in selected:
            row=dataset[i];spatial=row['public']['spatial'];base=spatial['base_logits'];grid=spatial['grid_world'];target=row['labels']['target_xy']
            network={k:v[None] for k,v in row['public']['network'].items()};result={}
            for control in ('A0',*CONTROLS):
                if control=='A0':mixed=base;updated=False
                else:
                    output=model(**controlled_inputs(network,row['meta'],control))
                    mixed,info=apply_revision(spatial,output,dict(enabled=True,scale=1.,threshold=0.,temperature=1.));updated=info['updated']
                ne=float((grid[mixed.argmax()]-target).norm());exact=torch.equal(base,mixed)
                if control in ('ambiguous','failed_detection') and not exact:raise ValueError('Missing evidence must return exact A0')
                t=totals[control];t['views']+=1;t['correct']+=int(ne<=20);t['ne_sum']+=ne;t['updated']+=int(updated)
                t['changed_peak']+=int(mixed.argmax()!=base.argmax());t['exact_A0']+=int(exact)
                result[control]=dict(NE=ne,correct=ne<=20,updated=updated)
            records.append(dict(target_instance=row['target_instance'],episode=row['episode'],step=row['step'],
                                valid_references=int(network['link_valid'].sum()),results=result))
    for t in totals.values():t.update(map_r1=t['correct']/t['views'],map_ne=t['ne_sum']/t['views'])
    report=dict(scope='Frozen TRAIN128 targets, one preselected cached teacher view each; uncalibrated scale1/threshold0; NOT navigation SR or gate selection',
                checkpoint_sha256=sha(checkpoint),controls=totals,records=records,
                caveats=['A0 prior remains language-conditioned in all controls; text suppression tests additional branch only.',
                         'no_geometry removes box-reference pair geometry only; public A0 context and candidate support remain available.',
                         'Link permutation is nontrivial only with multiple valid references.',
                         'Image-relative control rotates observed box-reference directions; language targets remain world directions.',
                         'Valid-target counterfactual-language diagnostic not yet implemented; no synthetic direction flip with unchanged GT.'])
    save(report,ROOT/'reports/a2_execution_20261027/REVISION_TRAIN_CONTROLS.json');print(json.dumps(totals),flush=True)


if __name__=='__main__':main()
