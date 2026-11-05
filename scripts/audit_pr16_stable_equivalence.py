"""CPU trace equivalence to pre-PR16 Stable Goal, including candidate context."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import torch
from graphnav.stable_navigation_goal import StableNavigationGoal,StableGoalConfig
from graphnav.full_localization import GoalField


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='d0b89ef');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    code=subprocess.check_output(['git','show',a.base+':graphnav/stable_navigation_goal.py'])
    module=types.ModuleType('_pr16_legacy_stable');sys.modules[module.__name__]=module
    exec(compile(code,'pre_PR16_stable_goal.py','exec'),module.__dict__)
    old=module.StableNavigationGoal();new=StableNavigationGoal()
    assert vars(old.config)==vars(new.config)
    torch.manual_seed(42);grid=torch.rand(1,16,2);events={};extra_reasons={}
    for step in range(200):
        logits=torch.randn(1,16);dense=torch.randn(1,16,128);ids=logits.topk(4,-1).indices
        forced=step%20 in (10,11)
        if forced:
            logits.fill_(-5);logits[0,(step//20+1)%16]=5
            ids=logits.topk(4,-1).indices
        field=GoalField(belief_logits=logits,candidate_ids=ids,
            candidate_xy=grid.gather(1,ids[...,None].expand(-1,-1,2)),
            candidate_context=dense.gather(1,ids[...,None].expand(-1,-1,128)),
            dense_context=dense,visual_coverage=(torch.rand(1,16)>.25).float())
        tracking=dict(state='tracking' if step%4 else 'geometry',visual_gain=1.0 if step%3 else 0.)
        if forced:
            tracking=dict(state='tracking',visual_gain=1.);field.visual_coverage.fill_(1)
        active=forced or step%5!=0
        x,xa=old.select(field,grid,tracking=tracking,visual_active=active)
        y,ya=new.select(field,grid,tracking=tracking,visual_active=active)
        for name in ('belief_logits','candidate_ids','candidate_xy','candidate_context','dense_context'):
            assert torch.equal(getattr(x,name),getattr(y,name)),(step,name)
        assert (old.goal_id,old.pending_id,old.pending_count,old.replans)==(new.goal_id,new.pending_id,new.pending_count,new.replans)
        events[ya['event']]=events.get(ya['event'],0)+1
        extra_reasons[ya['replan_block_reason']]=extra_reasons.get(ya['replan_block_reason'],0)+1
    assert new.replans>0
    result=dict(frames=200,all_fields_and_state_exactly_equal=True,thresholds=vars(new.config),
        events=events,diagnostic_reasons=extra_reasons,legacy_source_sha256=hashlib.sha256(code).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
