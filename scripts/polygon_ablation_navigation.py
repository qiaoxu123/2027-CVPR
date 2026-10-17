"""Same pretrained head, Guarded-Blend and shared evaluator for every geometry."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import time
import sys
from functools import lru_cache
import numpy as np
import torch

from scripts.polygon_ablation_plan import OUT
from scripts.polygon_ablation_engine import model_for, make_dataset, event
from graphnav.a3_runtime import A3TrajectoryPolicy, public_static_record
from graphnav.candidate_geometry import collate_full
from graphnav.polygon_ablation import geometry_view
from graphnav.full_experiment import _batch_inputs
from graphnav.trajectory_policy import CandidateTrajectoryStop
from graphnav.closedloop import PublicEpisode, PublicCatalog, ClosedLoopProtocol, rollout, score_trace, aggregate
from graphnav.checkpointing import file_sha256, atomic_json, resolve_checkpoint_resource
from graphnav.workspace_paths import PROJECT, WORKSPACE


@lru_cache(maxsize=6000)
def public_query_features(record_key,plan_sha256):
    """Reuse CPU geometry only, across seeds/arms; never cache a GT record.

    Same dataset conversion as the uncached path, keyed by the complete public
    record and immutable plan SHA. Views copy every tensor they change.
    """
    assert file_sha256(OUT/'frozen_plan.json')==plan_sha256
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    scratch=OUT/'public_inputs';scratch.mkdir(exist_ok=True)
    path=scratch/(hashlib.sha256(record_key.encode()).hexdigest()+'.jsonl')
    if path.exists():assert path.read_text()==record_key+'\n'
    else:path.write_text(record_key+'\n')
    dataset=make_dataset(plan,path)
    return dataset[0],dataset._prepared(0)[0]


class PolygonPolicy(A3TrajectoryPolicy):
    def __init__(self,plan,checkpoint,scope,representation,encoder,scratch,*,cache_public=True):
        self.plan=plan;self.args=SimpleNamespace(**plan['args']);self.device=torch.device('cuda')
        self.scope,self.representation=scope,representation
        self.cache_public=cache_public
        self.model=model_for(plan,frozen=checkpoint is None)
        if checkpoint:
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            self.model.load_state_dict(state['model'],strict=True)
        self.model.eval().requires_grad_(False)
        self.encoder=encoder;self.scratch=Path(scratch);self.scratch.mkdir(parents=True,exist_ok=True)
        state=torch.load(plan['navigation']['head_checkpoint'],map_location='cpu',weights_only=True)
        cache=resolve_checkpoint_resource(state['args']['cache'],plan['navigation']['head_checkpoint'])/'manifest.json'
        assert file_sha256(cache)==state['input_fingerprint']['cache_manifest']['sha256']
        assert json.loads(cache.read_text())['protocol']['checkpoint_sha256']==plan['sources']['reference']['sha256']
        assert state['protocol']['observed_fraction']==0.
        self.head=CandidateTrajectoryStop(hidden=state['hidden']).cuda().eval().requires_grad_(False)
        self.head.load_state_dict(state['model'],strict=True)
        self.stop_positive_weight=state['protocol']['stop_positive_weight'];self.stop_threshold=.3
        self.execution='guarded-blend';self.explore_candidates=0;self.fields={}

    def set_episode(self,episode,transform,catalog):
        self.record=public_static_record(episode,transform,catalog,max_landmarks=self.args.max_landmarks)
        self.visited_goal_cells=set()

    @torch.no_grad()
    def frozen_field(self):
        # Key contains ONLY the exact public static inputs, not episode IDs,
        # targets, future poses, or outcomes. No approximate text-only matching.
        key=json.dumps(self.record,sort_keys=True)
        if key not in self.fields:
            if self.cache_public:
                base,graph=public_query_features(key,file_sha256(OUT/'frozen_plan.json'))
            else:
                path=self.scratch/'current_public_query.jsonl';path.write_text(key+'\n')
                dataset=make_dataset(self.plan,path);base,graph=dataset[0],dataset._prepared(0)[0]
            item=geometry_view(base,graph,self.record['world_bounds'],
                         scope=self.scope,representation=self.representation,
                         prior_sigma_m=self.args.prior_sigma_m,rule_sigma_m=self.args.rule_sigma_m)
            batch=collate_full([item]);x,t,m,n=_batch_inputs(batch,self.encoder,self.device,'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                field=self.model(x,variant='geometry',tokens=t,token_mask=m,node_emb=n)
            field.dense_context=None
            self.fields[key]=field
        return self.fields[key],None


class PoseOnlyBuilder:
    """Elide unused rendering only after exact original-builder smoke parity."""
    def __init__(self,original):self.original=original
    def transform(self,map_name):return self.original.transform(map_name)
    def build_from_poses(self,episode,poses):return SimpleNamespace(pose=np.asarray(poses[-1]).copy())


@lru_cache(maxsize=1)
def audited_navigation_inputs():
    audit=json.loads((OUT/'navigation_data_audit.json').read_text())
    for row in audit['files']:
        assert file_sha256(row['path'])==row['sha256'], 'Navigation input changed: '+row['path']
    assert all(r['all_public_queries_match'] for r in audit['splits'].values())
    return audit


def resources(plan,split):
    audit=audited_navigation_inputs()
    source=PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'
    sys.path.insert(0,str(source))
    from sbfnav.config import load_config, apply_overrides
    from sbfnav.dataset import CityNavDataset, CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.mapdata import GROUND_LEVEL
    data=WORKSPACE/'DATA/sbfnav-data'
    cfg=apply_overrides(load_config(source/'configs/sbfnav/main.yaml'),['data.root='+str(data)])
    assert cfg['data']['version']==audit['navigation_loader_version']
    dataset=CityNavDataset(data,split,version=cfg['data']['version'])
    catalog=CityReferCatalog(data/'cityrefer')
    builder=PlanningStateBuilder(data,PublicCatalog(catalog),canvas_size=cfg['model']['canvas_size'],field_size=28,
                                max_height_above_ground_m=cfg['state']['max_height_above_ground_m'],trajectory_radius_m=cfg['state']['trajectory_radius_m'])
    protocol=ClosedLoopProtocol()
    controller=BoundedWaypointController(horizontal_step_m=50.,vertical_step_m=10.,minimum_above_ground_m=10.,maximum_above_ground_m=100.)
    return dataset,catalog,builder,controller,protocol,GROUND_LEVEL


def navigate(plan,encoder,job,scope,representation,checkpoint=None,split='val_seen',smoke=False,original_builder=False):
    if not (OUT/'all_static_complete.json').exists():
        raise ValueError('All seeds/representations must finish static evaluation before navigation')
    folder=OUT/'navigation'/job/(split+('_seen128' if smoke else '')+('_original_builder' if original_builder else ''))
    folder.mkdir(parents=True,exist_ok=True)
    data,catalog,base,controller,protocol,ground=resources(plan,split)
    from sbfnav.validation import evenly_spaced_indices
    indices=evenly_spaced_indices(len(data),128 if smoke else None)
    expected=128 if smoke else plan['navigation']['splits'][split]
    assert len(indices)==expected
    manifest=[dict(index=int(i),episode_id=list(data[i].episode_id),map_name=data[i].map_name) for i in indices]
    signature=dict(scope=scope,representation=representation,checkpoint_sha256=file_sha256(checkpoint) if checkpoint else plan['input_sha256'][str(__import__('scripts.polygon_ablation_plan',fromlist=['BASE']).BASE)],
                   head_sha256=plan['navigation']['head_sha256'],plan_sha256=file_sha256(OUT/'frozen_plan.json'),
                   navigation_data_audit_sha256=file_sha256(OUT/'navigation_data_audit.json'),
                   policy_source_sha256=file_sha256(__file__),
                   split=split,smoke=smoke,original_builder=original_builder,public_feature_cache=not original_builder)
    if (folder/'started.json').exists():
        assert json.loads((folder/'started.json').read_text())==signature
        assert json.loads((folder/'episodes.json').read_text())==manifest
    else:
        atomic_json(signature,folder/'started.json');atomic_json(manifest,folder/'episodes.json')
    if (folder/'metrics.json').exists():
        assert json.loads((folder/'metrics.json').read_text())['metrics']['episodes']==expected
        return folder
    trace=folder/'predictions.jsonl'
    done=[json.loads(l) for l in trace.read_text().splitlines()] if trace.exists() else []
    assert [r['episode_id'] for r in done]==[r['episode_id'] for r in manifest[:len(done)]]
    builder=base if original_builder else PoseOnlyBuilder(base)
    policy=PolygonPolicy(plan,checkpoint,scope,representation,encoder,folder/'public_inputs',cache_public=not original_builder)
    metrics=[r['metrics'] for r in done];started=time.time()
    with trace.open('a') as f:
        for position,i in enumerate(indices):
            if position<len(done):continue
            record=data[i]
            # Exactly the existing evaluator public interface, no target/future.
            public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
            policy.set_episode(public,builder.transform(public.map_name),catalog)
            poses,termination,timings,diagnostics=rollout(public,builder,controller,policy.predict,ground[public.map_name],protocol,synchronize=torch.cuda.synchronize)
            score=score_trace(poses,record.target_position,record.trajectory,termination,decision_seconds=timings)
            assert len(poses)-1<=20
            if len(poses)>1:
                delta=np.diff(poses,axis=0)
                assert np.linalg.norm(delta[:,:2],axis=1).max()<=50.+1e-4
                assert np.abs(delta[:,2]).max()<=10.+1e-4
            row=dict(index=int(i),episode_id=list(record.episode_id),map_name=record.map_name,poses=poses.tolist(),termination=termination,
                     decision_seconds=timings,step_diagnostics=diagnostics,metrics=score)
            f.write(json.dumps(row)+'\n');f.flush();metrics.append(score)
            if (position+1)%100==0:event(phase='navigation',job=job,split=split,smoke=smoke,completed=position+1,total=expected,seconds=time.time()-started)
    atomic_json(dict(status='Done',signature=signature,metrics=aggregate(metrics),seconds=time.time()-started,protocol=protocol.__dict__,
                     controller_transfer_limitation='Same unchanged pretrained head for all arms; newly trained candidate contexts may shift latent distribution. No per-representation controller adaptation.',
                     pose_rendering='Original observation builder' if original_builder else 'Pose-only rendering elision; original-builder Seen128 exact trace parity required'),folder/'metrics.json')
    del policy
    torch.cuda.empty_cache()
    return folder


def verify_rendering_parity(first,second):
    a=[json.loads(l) for l in (first/'predictions.jsonl').read_text().splitlines()]
    b=[json.loads(l) for l in (second/'predictions.jsonl').read_text().splitlines()]
    assert len(a)==len(b)==128
    for x,y in zip(a,b):
        assert x['episode_id']==y['episode_id'] and x['termination']==y['termination']
        assert np.array_equal(np.asarray(x['poses']),np.asarray(y['poses']))
        for key in ('SR','SPL','OSR','NE','false_stop'):
            assert x['metrics'][key]==y['metrics'][key]
    atomic_json(dict(samples=128,all_trajectories_exact=True,all_terminations_exact=True,all_navigation_metrics_exact=True,
                     original_reference='Original renderer and uncached public geometry conversion',optimized='Pose-only renderer and exact-public-record CPU feature cache'),OUT/'rendering_parity.json')
