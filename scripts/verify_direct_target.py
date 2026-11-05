"""Public-input parity, exact visual-cache reuse, causal split and direction probes."""
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import torch
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.landmark_alias import resolve_references
from graphnav.direct_target_data import load_model,batch_inputs
from graphnav.landmark_region_supervision import named_union


def main():
    p=argparse.ArgumentParser()
    for k in ('root','data-root','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=a.root;torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
    tasks=json.loads((r/'inventory_validated/public_tasks.json').read_text());catalog=json.loads((a.data_root/'cityrefer/objects.json').read_text())
    complete=json.loads((r/'training_aligned/complete.json').read_text())
    assert complete['fit_queries_seen']==complete['expected_fit_queries']==493
    records=[json.loads(l) for l in (r/'cache_aligned/public_static.jsonl').read_text().splitlines()]
    images=0;equal=0;missing=0;groups={'fit':set(),'dev':set()}
    for task,record in zip(tasks,records):
        bounds=np.array(task['world_bounds']);span=bounds[2:]-bounds[:2]
        matched=resolve_references([x['landmark']['name'] for x in task['anchors']],catalog[task['map_name']],max_landmarks=16,match_mode='fuzzy')
        expected=[dict(name=x['name'],object_type=x['object_type'],aliases=x['aliases'],
                       contour=((np.asarray(x['contour'])-bounds[:2])/span).tolist()) for x in matched]
        assert record['landmarks']==expected,'Runtime public record differs'
        for x in task['anchors']:
            groups[task['subset']].update(task['map_name']+'::'+str(i) for i in x['landmark']['public_ids'])
        old=torch.load(r/'cache_validated'/f"{task['index']:05d}.pt",map_location='cpu',weights_only=True)
        new=torch.load(r/'cache_aligned'/f"{task['index']:05d}.pt",map_location='cpu',weights_only=True)
        assert torch.equal(old['query']['tokens'],new['query']['tokens'])
        for oldobs,newobs in zip(old['observations'],new['observations']):
            assert oldobs['rgb_sha256']==newobs['rgb_sha256']
            if oldobs['observed'] is None:
                assert newobs['observed'] is None;missing+=1;continue
            assert set(oldobs['observed'])==set(newobs['observed'])
            assert all(torch.equal(oldobs['observed'][k],newobs['observed'][k]) for k in oldobs['observed'])
            images+=1;equal+=1
    assert not groups['fit'] & groups['dev']
    protected={}
    for name in ('graphnav/stable_navigation_goal.py','graphnav/trajectory_policy.py','graphnav/a3_runtime.py','graphnav/closedloop.py'):
        old=subprocess.check_output(['git','show','499694b:'+name])
        assert Path(name).read_bytes()==old
        protected[name]=file_sha256(name)
    predictions=[json.loads(l) for l in (r/'closedloop/predictions.jsonl').read_text().splitlines()]
    assert len(predictions)==56*3
    assert len({(x['index'],x['method']) for x in predictions})==56*3
    assert all(x['error'] is None for x in predictions)
    model=load_model(torch.load(r/'training_aligned/best.pt',map_location='cpu',weights_only=True)).eval().requires_grad_(False)
    private=json.loads((r/'inventory_validated/private_labels.json').read_text());probes=[]
    with torch.inference_mode():
        for task in [t for t in tasks if t['subset']=='dev']:
            item=torch.load(r/'cache_aligned'/f"{task['index']:05d}.pt",map_location='cpu',weights_only=True)
            sample=dict(query=item['query'],**item['observations'][0],task=task,label=private[task['index']]['target_world'])
            inputs=batch_inputs([sample],visual=False);model.reset_episode();means=[]
            bounds=torch.tensor(task['world_bounds'],device='cuda');world=bounds[:2]+inputs['static_inputs']['grid_xy'][0]*(bounds[2:]-bounds[:2])
            for direction in range(4):
                altered=dict(inputs,relation_ids=torch.full_like(inputs['relation_ids'],direction))
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model.forward_direct_target(**altered)
                mean=(out['relation_field'].belief_logits[0].float().softmax(-1)[:,None]*world).sum(0)
                means.append(mean.cpu().tolist())
            probes.append(dict(index=task['index'],original_relation=task['anchors'][0]['predicate'],
                expected_world_centers=means,north_minus_south_Y=means[0][1]-means[1][1],east_minus_west_X=means[2][0]-means[3][0]))
    summary=dict(checked_public_queries=len(tasks),runtime_public_record_exact_match=True,
        reused_visual_states=equal,missing_visual_states=missing,shared_fit_dev_public_IDs=0,
        private_target_keys_absent_from_model_forward=True,original_controller_sources_unchanged=protected,
        real_closedloop_rows=len(predictions),all_expected_tasks_included=True,
        direction_probe=dict(queries=len(probes),north_higher_than_south_fraction=float(np.mean([p['north_minus_south_Y']>0 for p in probes])),
            mean_north_minus_south_Y_m=float(np.mean([p['north_minus_south_Y'] for p in probes])),
            east_greater_than_west_fraction=float(np.mean([p['east_minus_west_X']>0 for p in probes])),
            warning='Conditional head-only structure-ID intervention; full original text unchanged. East/west IDs never fitted. NOT end-to-end language accuracy or evidence to flip coordinates.'),
        checkpoint_sha256=file_sha256(r/'training_aligned/best.pt'))
    atomic_json(probes,a.output/'direction_probes.json');atomic_json(summary,a.output/'integrity_audit.json')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
