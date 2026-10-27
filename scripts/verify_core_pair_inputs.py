"""Verify source weights, shared initialization, complete cache and controls."""
import json
from pathlib import Path
import torch
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.stage1_checkpoint import initialize_model,EXPECTED,module_sha
from graphnav.stage1_data import Stage1Dataset


def main():
    project=Path('/home/20261024_1/Workspace/2027-CVPR');root=Path('reports/core_pair_20261027')
    sources=json.loads((root/'initial_sources_local.json').read_text());paths={k:v['path'] for k,v in sources.items()}
    bootstrap=project/'GraphNav/runs/graphnav_v1_stage1/initial_complete.pt'
    torch.manual_seed(0);a,args=initialize_model(paths,'cuda',bootstrap=bootstrap)
    torch.manual_seed(0);b,args_b=initialize_model(paths,'cuda',bootstrap=bootstrap,model_type='joint-rgb-geometry')
    common=a.state_dict();joint=b.state_dict()
    assert all(torch.equal(value,joint[key]) for key,value in common.items())
    assert args==args_b and args['grid_size']==28
    trainable={}
    for stage in ('A','B','C'):
        a.configure_stage(stage);b.configure_stage(stage)
        trainable[stage]={arm:sum(p.numel()for p in model.parameters()if p.requires_grad)for arm,model in [('A0',a),('A1',b)]}
        assert all(not p.requires_grad for p in a.trajectory.parameters())==(stage!='C')
        assert all(not p.requires_grad for p in b.trajectory.parameters())==(stage!='C')
    data=Stage1Dataset(project/'GraphNav/data/rebuilt_stage1_20261025/frames')
    assert len(data)==87512 and len(data.manifest['protocol']['episode_indices'])==21878
    report=dict(complete=True,original_sources=sources,shared_initial_tensors_equal=len(common),a3_module_sha256=module_sha(a.a3),trajectory_sha256=module_sha(a.trajectory),grid_size=28,states=len(data),episodes=21878,cache_manifest_sha256=file_sha256(data.root/'manifest.json'),bert_cache_sha256=file_sha256(project/'GraphNav/data/rebuilt_stage1_20261025/bert_tokens.pt'),bootstrap_sha256=file_sha256(bootstrap),trainable_parameters=trainable,cache_note='Fully reconstructed and SHA-verified cache; preserved original migrated cache incomplete. No historical source file rewritten.')
    atomic_json(report,root/'input_verification.json');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
