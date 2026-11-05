"""Freeze final fixed-budget PR23 weights and inference code before Val-Seen."""
import argparse,json
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json,file_sha256,provenance


def main():
    p=argparse.ArgumentParser()
    for k in ('run','stage1','original-v2','existing-full','config','public-objects'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();train=a.run/'train_landmark_relation';s=torch.load(train/'last.pt',weights_only=True,map_location='cpu');assert not s['smoke'] and s['completed_epochs']==s['planned_epochs']==5;assert all(x['observed']==21878 for x in s['history'])
    sources=dict(stage1=a.stage1,original_v2=a.original_v2,existing_full=a.existing_full,trained_landmark_relation=train/'last.pt',config=a.config,public_objects=a.public_objects)
    code=['graphnav/unified_pgvb.py','graphnav/full_localization.py','graphnav/aerialvg_inspired_relation.py','graphnav/unified_full_relation_runtime.py','graphnav/unified_text_ablation.py','graphnav/unified_text_ablation_runtime.py','graphnav/a3_runtime.py','graphnav/stable_navigation_goal.py','graphnav/trajectory_policy.py','scripts/prepare_unified_v2_landmark_relation_ablation.py','scripts/evaluate_closedloop.py']
    payload=dict(sources={k:dict(path=str(v),SHA256=file_sha256(v)) for k,v in sources.items()},code={k:file_sha256(k) for k in code},provenance=provenance(),training=dict(seed=42,epochs=5,records_per_epoch=21878,checkpoint_selection='fixed final epoch; no Val/Dev selection',cancelled_relation_only=True,retrained_full_text=False),evaluation=dict(split='val_seen',expected=2470,methods=['full_text','landmark_relation'],visual_control='off',public_reference_binding='same full public catalog and exact text matching',horizon=20,success_radius_m=20.,horizontal_step_m=50.,stop_threshold=.3,arrival_guard_m=.001),Test_authorized=False)
    dest=a.run/'FROZEN_BEFORE_VAL.json'
    if dest.exists():assert json.load(open(dest))==payload,'Frozen sources/code changed'
    else:atomic_json(payload,dest)
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
