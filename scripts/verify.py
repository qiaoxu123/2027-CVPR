"""Real data backward, structural probes, zero-field parity and controller routing."""
from common import *
import argparse
import numpy as np
import torch
from unittest.mock import patch
from model import LocalRelationNetwork
from labels import objective
from train import load_groups,audit_groups,labels_on
from schema import to_device
from policy import RefinementPolicy,updated_field
from graphnav.stage1_runtime import GraphNavStage1Policy
from graphnav.a3_runtime import A3TrajectoryPolicy
from graphnav.dynamic_runtime import NativeObservationBuilder
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout
from sbfnav.dataset import CityNavDataset,CityReferCatalog
from sbfnav.planning_state import PlanningStateBuilder
from sbfnav.navigation import BoundedWaypointController
from sbfnav.mapdata import GROUND_LEVEL
from prepare import DetailBuilder,causal_extract


def main():
    p=argparse.ArgumentParser();p.add_argument('--skip-parity',action='store_true');p.add_argument('--allow-partial',action='store_true');p.add_argument('--check-resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);c=configuration();verify_sources(c);groups=load_groups(args.allow_partial)
    rows=groups['train']+groups['dev'];row=next((r for r in rows if len(r['meta']['candidates']) and len(r['meta']['references'])),rows[0])
    torch.manual_seed(17);model=LocalRelationNetwork(rgb_dim=row['public']['network']['candidate_rgb'].shape[-1]).cuda().eval()
    x=to_device(row['public'],'cuda');y=labels_on(row,'cuda');out=model(**x);loss,_=objective(out,y,True);loss.backward()
    grads={k:float(v.grad.norm()) for k,v in model.named_parameters() if v.grad is not None}
    assert torch.isfinite(loss) and any(v>0 for v in grads.values())
    with torch.no_grad():
        a=out['scores'];t={k:v.clone() for k,v in x.items()};t['tokens']=-t['tokens'];language=float((a-model(**t)['scores']).abs().max()) if len(a) else 0.
        t={k:v.clone() for k,v in x.items()};t['pair'][...,4]+=1;distance=float((a-model(**t)['scores']).abs().max()) if len(a) else 0.
    result=dict(scope='implementation verification, not trained model efficacy',real_views=sum(len(v) for v in groups.values()),coverage=audit_groups(groups),
        real_loss=float(loss.detach()),trainable_parameters=sum(p.numel() for p in model.parameters()),nonzero_gradient_tensors=sum(v>0 for v in grads.values()),
        language_structural_score_change=language,distance_structural_score_change=distance,source_hashes_verified=True,parity=[])
    if args.check_resume:
        train=CityNavDataset(DATA,'train_seen',version=c['data_version']);cat=CityReferCatalog(DATA/'cityrefer')
        first=groups['train'][0]['episode'];episode_rows=sorted([r for r in groups['train'] if r['episode']==first],key=lambda r:r['step'])
        rec=train[first];ep=PublicEpisode(rec.map_name,rec.instruction,rec.start_pose.copy(),tuple(cat.referenced_landmarks(rec)),tuple(cat.referenced_landmark_names(rec)))
        builder=DetailBuilder(PlanningStateBuilder(DATA,PublicCatalog(cat),canvas_size=384,field_size=28),RGBD,ROOT/'runs/resume_render')
        adapter=RefinementPolicy(ROOT/'runs/resume_query',collect=True);adapter.set_episode(ep,builder.transform(rec.map_name),cat)
        steps=[r['step'] for r in episode_rows];cached={steps[0]:episode_rows[0]['public']};calls=[]
        def compute(step):
            calls.append(step);state=builder.build_from_poses(ep,np.asarray(rec.trajectory)[:step+1]);adapter.predict(state,rec.instruction)
            return {g:{k:v.detach().cpu().clone() for k,v in values.items()} for g,values in adapter.pack.items()}
        restored=dict(causal_extract(steps,cached,compute));comparisons=[]
        for r in episode_rows:
            same=all(torch.allclose(restored[r['step']][g][k],r['public'][g][k],atol=1e-5,rtol=0) for g in r['public'] for k in r['public'][g])
            assert same;comparisons.append(dict(step=r['step'],public_matches_uninterrupted=True))
        assert calls==steps
        result['real_partial_resume']=dict(episode=first,replayed_steps=calls,comparisons=comparisons)
    if not args.skip_parity:
        ds=CityNavDataset(DATA,'val_seen',version=c['data_version']);cat=CityReferCatalog(DATA/'cityrefer')
        builder=NativeObservationBuilder(PlanningStateBuilder(DATA,PublicCatalog(cat),canvas_size=384,field_size=28),RGBD,ROOT/'runs/parity_render')
        original=GraphNavStage1Policy(CHECKPOINT,ROOT/'runs/parity_original');adapter=RefinementPolicy(ROOT/'runs/parity_adapter')
        controller=BoundedWaypointController(horizontal_step_m=50.,vertical_step_m=10.,minimum_above_ground_m=10.,maximum_above_ground_m=100.)
        archived_path=SOURCE/'runs/graphnav_v1_stage1/full_seen/predictions.jsonl'
        if archived_path.exists():
            with archived_path.open() as f:archived=[json.loads(next(f)) for _ in range(2)]
        else:archived=json.loads((ROOT/'assets/paper_parity_traces.json').read_text())
        for i in [0,1]:
            rec=ds[i];ep=PublicEpisode(rec.map_name,rec.instruction,rec.start_pose.copy(),tuple(cat.referenced_landmarks(rec)),tuple(cat.referenced_landmark_names(rec)))
            traces=[];terms=[]
            for policy in [original,adapter]:
                policy.set_episode(ep,builder.transform(rec.map_name),cat)
                poses,term,_,_=rollout(ep,builder,controller,policy.predict,GROUND_LEVEL[rec.map_name],ClosedLoopProtocol(),synchronize=torch.cuda.synchronize)
                traces.append(poses);terms.append(term)
            equal=traces[0].shape==traces[1].shape and np.allclose(traces[0],traces[1],atol=1e-5,rtol=0) and terms[0]==terms[1]
            old=np.asarray(archived[i]['poses']);paper=old.shape==traces[0].shape and np.allclose(old,traces[0],atol=1e-4,rtol=0)
            assert equal and paper
            result['parity'].append(dict(index=i,poses=len(traces[0]),matches_original=True,matches_paper_archive=True))
        rec=ds[0];ep=PublicEpisode(rec.map_name,rec.instruction,rec.start_pose.copy(),tuple(cat.referenced_landmarks(rec)),tuple(cat.referenced_landmark_names(rec)))
        adapter.set_episode(ep,builder.transform(rec.map_name),cat);state=builder.build_from_poses(ep,np.asarray([ep.start_pose]));before=adapter.predict(state,rec.instruction)
        def forced(module,args,out):
            logits=out.belief.belief_logits[0].float().clone();alternate=int(out.belief.candidate_ids[0,1]);logits[alternate]=float(logits.max()+20)
            out.belief=updated_field(out.belief,adapter.inputs['grid_xy'],logits);adapter.corrected_field=out.belief
        hook=adapter.model.register_forward_hook(forced);adapter.set_episode(ep,builder.transform(rec.map_name),cat)
        real=A3TrajectoryPolicy.predict_field;calls=[]
        def counted(pol,st,field):calls.append(field);return real(pol,st,field)
        with patch.object(A3TrajectoryPolicy,'predict_field',counted):after=adapter.predict(state,rec.instruction)
        hook.remove();assert len(calls)==1 and not np.allclose(before.diagnostics['selected_goal_xy'],after.diagnostics['selected_goal_xy'])
        result.update(single_controller_call=True,forced_diagnostic_field_changes_executed_goal=True)
    save(result,ROOT/'reports/verification.json');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
