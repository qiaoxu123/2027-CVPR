"""Verify collision consequences by frozen GPU forward, TRAIN posthoc only."""
import argparse
import json
from pathlib import Path
import torch
from graphnav.checkpointing import file_sha256
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_retarget_grounding import attach_retarget_data,collate
from scripts.train_local_chain_position import records,batch_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--collisions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior audit')
    torch.set_num_threads(4)
    cfg=json.loads(a.config.read_text())['config']
    roles=torch.load(cfg['role_vectors'],map_location='cpu',weights_only=True)
    allowed=set(json.loads(Path(cfg['filter']).read_text())['frame_keys'])
    base=records(Path(cfg['cache']),Path(cfg['fit_labels']),'fit',roles,'reference',allowed)
    all_rows,_=attach_retarget_data(base,Path(cfg['fit_labels']),Path(cfg['teacher']),Path(cfg['public_inputs']))
    indexed={(row.get('source_frame_key'),row['task_key']):row for row in all_rows if row.get('source_frame_key')}
    pairs=json.loads(a.collisions.read_text())['cases'];results={}
    for name,path in [('frozen',Path(cfg['warm_start'])),('soft_instance',Path(cfg['output'])/'final.pt')]:
        model=WorldLikelihoodPositionHead().cuda().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
        observations=[]
        for case in pairs:
            rows=[indexed[(case['source_frame_key'],key)] for key in case['task_keys']]
            batch=collate(rows);public=batch_inputs(batch,'cuda')
            with torch.no_grad():
                _,visual=model(**public,return_visual=True)
            delta=float((visual[0]-visual[1]).abs().max())
            assert delta<1e-6, 'Expected identical visual logits for exact local text collision'
            valid=public['local_valid'];top=visual.masked_fill(~valid,float('-inf')).argmax(-1).cpu()
            matches=[bool(row['weak_instance_labels'][top[i]]==1) for i,row in enumerate(rows)]
            assert not all(matches),'Disjoint weak target cores cannot share a visual Top1'
            observations.append(dict(source_frame_key=case['source_frame_key'],task_keys=case['task_keys'],
                max_abs_visual_score_difference=delta,top1_indices=top.tolist(),weak_core_top1_matches=matches,
                world_goal_can_differ=True,warning='weak core matches only, not verified instance recall'))
        results[name]=dict(checkpoint_sha256=file_sha256(path),cases=observations,
            matched_weak_targets=sum(sum(x['weak_core_top1_matches']) for x in observations),weak_targets=2*len(observations))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(results=results,source_sha256=file_sha256(__file__),
        gt_used_only_for_match_scoring=True,forward_inputs=list(public)),indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
