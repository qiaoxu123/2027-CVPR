"""Explicit new-static/old-controller transfer; preserved baseline runtime is untouched."""
import json
from types import SimpleNamespace
import torch
from .a3_runtime import A3WaypointPolicy,A3TrajectoryPolicy
from .candidate_geometry import FullCandidateDataset,collate_full
from .full_experiment import _batch_inputs
from .static_controls import graph_control


COMPATIBLE_KEYS=('hidden','heads','grid_size','top_k','max_landmarks','max_roads','max_relations','prior_sigma_m','rule_sigma_m','rule_weight','relative_position')


def validate_static_transfer(candidate,reference):
    for key in COMPATIBLE_KEYS:
        default=False if key=='relative_position' else None
        if candidate.get(key,default)!=reference.get(key,default):
            raise ValueError('static/controller transfer configuration differs: '+key)


class StaticTransferTrajectoryPolicy(A3TrajectoryPolicy):
    def __init__(self,checkpoint,head_checkpoint,reference_checkpoint,scratch_directory,*,device='cuda',stop_threshold=.3,execution='guarded-blend'):
        if execution!='guarded-blend':raise ValueError('controlled static transfer requires original guarded-blend execution')
        candidate=torch.load(checkpoint,map_location='cpu',weights_only=True)
        reference=torch.load(reference_checkpoint,map_location='cpu',weights_only=True)
        validate_static_transfer(candidate['args'],reference['args'])
        if candidate['args'].get('max_train_batches'):raise ValueError('Smoke checkpoint cannot be final navigation candidate')
        self.static_variant=candidate.get('variant','geometry')
        if self.static_variant not in ('geometry','gat'):raise ValueError('static-only checkpoint required')
        self.static_control=candidate['args'].get('graph_control','full')
        A3WaypointPolicy.__init__(self,checkpoint,scratch_directory,device=device)
        # Verify the OLD controller against its ORIGINAL static feature manifest.
        # Then explicitly transfer its unchanged parameters to a compatible field.
        self.load_trajectory_head(reference_checkpoint,head_checkpoint,stop_threshold=stop_threshold,execution=execution,explore_candidates=0)

    @torch.no_grad()
    def frozen_field(self):
        key=json.dumps(self.record,sort_keys=True)
        if key not in self.field_cache:
            self.field_cache.clear()
            path=self.scratch/'current_public_query.jsonl';path.write_text(key+'\n')
            dataset=FullCandidateDataset(path,grid_size=self.args.grid_size,max_landmarks=self.args.max_landmarks,max_roads=self.args.max_roads,max_relations=self.args.max_relations,prior_sigma_m=self.args.prior_sigma_m,rule_sigma_m=self.args.rule_sigma_m)
            inputs,tokens,mask,names=_batch_inputs(collate_full([dataset[0]]),self.encoder,self.device,self.static_variant)
            inputs=graph_control(inputs,self.static_control)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                field=self.model(inputs,variant=self.static_variant,tokens=tokens,token_mask=mask,node_emb=names)
            self.field_cache[key]=(field,inputs)
            self.public_node_emb=names;self.public_entities=dataset._prepared(0)[0]['entities']
        return self.field_cache[key]
