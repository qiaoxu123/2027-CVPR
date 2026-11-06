"""Explicitly privileged Oracle executable. Never import in normal evaluator.

This wrapper replaces only the Belief field before existing StableGoal and
Trajectory/Stop. It receives precomputed angles, never target positions/ranges.
"""
import argparse,json,sys,hashlib
from pathlib import Path
import torch
from scripts.oracle_bearing_geometry import rerank,replace_field
from scripts.prepare_oracle_bearing import key
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--oracle-arm',choices=['B','C','D1'],required=True);p.add_argument('--oracle-angles',type=Path,required=True)
    a,rest=p.parse_known_args();assert '--split' in rest and rest[rest.index('--split')+1]=='val_seen'
    assert '--instruction-mode' in rest and rest[rest.index('--instruction-mode')+1]=='full_text'
    assert '--relation-visual-control' in rest and rest[rest.index('--relation-visual-control')+1]=='off'
    out=Path(rest[rest.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads(a.oracle_angles.read_text());assert frozen['ORACLE'] and frozen['GT_range_supplied'] is False
    import graphnav.unified_text_ablation_runtime as adapter
    original=adapter.UnifiedTextAblationPolicy
    class OracleOnlyDiagnosticPolicy(original):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw)
            actual=self.unified.forward_full_relation
            def forward(**inputs):
                result=actual(**inputs);field=result['field'];entry=self.oracle_entry
                if entry['angles']:
                    b=torch.tensor(self.record['world_bounds'],device=self.device,dtype=torch.float32);grid=inputs['static_inputs']['grid_xy'];world=grid[0]*(b[2:]-b[:2])+b[:2]
                    z=rerank(field.belief_logits[0],world,entry['anchors'],entry['angles'],a.oracle_arm[0],cell_diagonal=float(((b[2:]-b[:2])/28).norm()))[None]
                    corrected=replace_field(field,z,grid);result=dict(result,field=corrected)
                    self.oracle_change_count+=int(corrected.candidate_ids[0,0]!=field.candidate_ids[0,0])
                return result
            self.unified.forward_full_relation=forward
        def set_episode(self,episode,transform,catalog):
            super().set_episode(episode,transform,catalog)
            k=key(episode.map_name,episode.instruction,episode.start_pose)
            self.oracle_entry=frozen['tasks'][k];self.oracle_change_count=0
            assert self.oracle_entry['public_binding_SHA256']==hashlib.sha256(json.dumps(self.record['landmarks'],sort_keys=True).encode()).hexdigest()
            assert self.oracle_entry['landmark_ids']==[r['id'] for r in self.record['landmarks']]
        def predict(self,state,instruction):
            out=super().predict(state,instruction)
            out.diagnostics.update(PRIVILEGED_ORACLE_DIAGNOSTIC=True,oracle_arm=a.oracle_arm,oracle_angles=self.oracle_entry['angles'],oracle_group=self.oracle_entry['group'],oracle_changed_top1_steps=self.oracle_change_count,GT_range_or_position_at_runtime=False)
            return out
    adapter.UnifiedTextAblationPolicy=OracleOnlyDiagnosticPolicy
    lock=dict(ORACLE=True,arm=a.oracle_arm,angles_SHA=file_sha256(a.oracle_angles),wrapper_SHA=file_sha256(__file__),geometry_SHA=file_sha256(Path('scripts/oracle_bearing_geometry.py')),baseline_checkpoint_required='9ac51ee253eed46acbf66e6afb47888857b70258f5fcd195721723d5112dc387')
    ck=rest[rest.index('--relation-checkpoint')+1];assert file_sha256(ck)==lock['baseline_checkpoint_required']
    f=out/'ORACLE_MANIFEST.json'
    if f.exists():assert json.loads(f.read_text())==lock
    else:f.write_text(json.dumps(lock,indent=2))
    from scripts import evaluate_closedloop
    sys.argv=['isolated_oracle_diagnostic',*rest];evaluate_closedloop.main()
    m=out/'metrics.json';d=json.loads(m.read_text());d.update(PRIVILEGED_ORACLE_DIAGNOSTIC=True,deployable=False,oracle_arm=a.oracle_arm);m.write_text(json.dumps(d,indent=2))
if __name__=='__main__':main()
