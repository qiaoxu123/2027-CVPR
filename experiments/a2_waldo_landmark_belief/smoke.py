"""Serial CUDA smoke on actual TRAIN observations; not an efficacy experiment."""
import gc
import json
import time
import numpy as np
import torch
from .paths import bootstrap, ROOT, DATA, RGBD, CHECKPOINT, save
bootstrap()
from .policy import A2Policy
from .model import LandmarkBeliefNetwork,supervision_loss
from .private_labels import PrivateAssociationIndex,label_view
from graphnav.stage1_runtime import GraphNavStage1Policy
from graphnav.closedloop import PublicEpisode,PublicCatalog
from graphnav.dynamic_runtime import NativeObservationBuilder
from sbfnav.dataset import CityNavDataset,CityReferCatalog
from sbfnav.planning_state import PlanningStateBuilder
from sbfnav.mapdata import GROUND_LEVEL


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started=time.perf_counter()
    manifest=json.loads((ROOT/'runs/a2_waldo_landmark_20261027/source_manifest.json').read_text())
    chosen=next(e for e in manifest['entries'] if e['split']=='train' and e['linked_references']>=1)
    ds=CityNavDataset(DATA,'train_seen')
    cat=CityReferCatalog(DATA/'cityrefer')
    record=ds[chosen['index']]
    episode=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),
                          tuple(cat.referenced_landmarks(record)),tuple(cat.referenced_landmark_names(record)))
    builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(cat),canvas_size=384,field_size=28),
                                     RGBD,ROOT/'runs/a2_waldo_landmark_20261027/smoke/render')
    trajectory=np.asarray(record.trajectory)
    legal=np.flatnonzero((trajectory[:,2]-GROUND_LEVEL[record.map_name]>=10) & (trajectory[:,2]-GROUND_LEVEL[record.map_name]<=100))
    steps=sorted({int(legal[round(q*(len(legal)-1))]) for q in (.25,.75,1.)})
    states=[builder.build_from_poses(episode,trajectory[:step+1]) for step in steps]
    baseline=GraphNavStage1Policy(CHECKPOINT,ROOT/'runs/a2_waldo_landmark_20261027/smoke/a0')
    baseline.set_episode(episode,builder.transform(record.map_name),cat)
    baseline_predictions=[baseline.predict(s,record.instruction) for s in states]
    del baseline
    gc.collect();torch.cuda.empty_cache()
    adapter=A2Policy(ROOT/'runs/a2_waldo_landmark_20261027/smoke/a2',collect=True)
    adapter.set_episode(episode,builder.transform(record.map_name),cat)
    index=PrivateAssociationIndex(cat)
    model=LandmarkBeliefNetwork().cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    rows=[]
    for state,before in zip(states,baseline_predictions):
        tick=time.perf_counter()
        after=adapter.predict(state,record.instruction)
        torch.cuda.synchronize()
        parity=bool(np.allclose(before.xyz,after.xyz,atol=1e-5,rtol=0) and before.should_stop==after.should_stop
                    and before.diagnostics['belief_top16_world_xy']==after.diagnostics['belief_top16_world_xy'])
        assert parity and adapter.corrected_field is adapter.original_field
        pack=adapter.pack
        labels,private=label_view(record,cat,index,pack,adapter.meta,state.current_observation,'train')
        out=model(**{k:v.detach().float()[None] if v.dtype.is_floating_point else v[None] for k,v in pack['network'].items()})
        loss,stats=supervision_loss(out,{k:v.cuda()[None] for k,v in labels.items()},refine=True)
        optimizer.zero_grad();loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
        optimizer.step()
        assert not any(p.requires_grad for p in adapter.model.parameters())
        assert not any(p.requires_grad for p in adapter.bert.model.parameters())
        assert not any(p.requires_grad for p in adapter.native.model.parameters())
        assert not any(p.requires_grad for p in adapter.perception.detector.model.parameters())
        row=dict(public={g:{k:v.detach().cpu().clone() for k,v in values.items()} for g,values in pack.items()},
                 labels=labels,meta=adapter.meta,private=private,episode=chosen['index'],step=state.step)
        torch.save(row,ROOT/f'runs/a2_waldo_landmark_20261027/smoke/view_{state.step}.pt')
        rows.append(dict(step=state.step,a0_exact_prediction_parity=parity,candidates=len(adapter.meta['candidates']),
                         references=adapter.meta['valid_references'],loss=float(loss.detach()),**stats,
                         detector_error=adapter.meta['detector_error'],seconds=time.perf_counter()-tick))
        print(rows[-1],flush=True)
    save(dict(status='cuda_smoke_passed',scope='Actual train observations and optimizer steps, not full training or efficacy',
              episode=chosen['index'],map_name=record.map_name,rows=rows,all_backbones_frozen=True,
              trainable_parameters=sum(p.numel() for p in model.parameters()),
              max_cuda_memory_gb=torch.cuda.max_memory_allocated()/2**30,seconds=time.perf_counter()-started),
         ROOT/'reports/a2_execution_20261027/cuda_smoke.json')


if __name__=='__main__':
    main()
