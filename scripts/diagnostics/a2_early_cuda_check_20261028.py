"""Provisional real A2 CUDA diagnostic; never issues a formal training gate."""
import argparse,json,fcntl,time
from pathlib import Path
import torch
from experiments.a2_waldo_landmark_belief.paths import ROOT,DATA,sha,save
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.relation_v3_data import read_training_source,build_training_row
from experiments.a2_waldo_landmark_belief.relation_v3_train import initial_model,fingerprint,code_hashes,require_idle_gpu,optimizer_for,PARENT
from experiments.a2_waldo_landmark_belief.relation_v3_loss import relation_v3_loss
from experiments.a2_waldo_landmark_belief.batching import collate_views,to_device
from scripts.a2_prepare_relation_v3_roles import verify_target_roles


def run(expected):
    directory=BASE/'revision_v3/remote_migration/early_cuda';selection=directory/'selection.json';output=directory/'smoke_result.json'
    if output.exists():raise ValueError('Inspect an existing diagnostic; do not repeat it blindly')
    if sha(selection)!=expected:raise ValueError('Approved source selection changed')
    manifest=json.loads(selection.read_text())
    if (manifest['source_role']!='train' or manifest['calibration_rows_allowed'] is not False
        or manifest['formal_training'] is not False or manifest['scope']!='provisional_real_A2_CUDA_check_only_not_formal_gate'):raise ValueError('Only the approved TRAIN-only diagnostic is allowed')
    if code_hashes()!=manifest['code_sha256']:raise ValueError('Reviewed model/policy source differs')
    roles_path=BASE/'revision_v3/roles.json'
    if sha(roles_path)!=manifest['roles_sha256']:raise ValueError('Full original roles changed')
    roles=json.loads(roles_path.read_text());groups={name:{(r['map_name'],r['object_id']) for r in entries} for name,entries in roles['target_roles'].items()};verify_target_roles(groups)
    allowed=groups['train'];forbidden=groups['dev']|groups['calibration_fit']|groups['calibration_verify']
    if sha(DATA/'cityrefer/objects.json')!=manifest['public_objects_sha256']:raise ValueError('Original category annotations changed')
    objects=json.loads((DATA/'cityrefer/objects.json').read_text());rows=[];counts={};seen=set()
    families={'teacher_snapshot':BASE/'collected_snapshot','revision_v1_A0_closedloop':BASE/'revision_v1/confirmation','revision_v2_A0_closedloop':BASE/'revision_v2/confirmation'}
    if len(manifest['sources'])!=6:raise ValueError('Exactly the original six source views required')
    for selected in manifest['sources']:
        entry=selected['reference'];target=(entry['map_name'],entry['object_id']);family=entry['source_kind']
        if (entry['role']!='train' or target not in allowed or target in forbidden or entry not in roles['rows']
            or family not in families or Path(entry['directory']).resolve()!=families[family].resolve()):raise ValueError('Calibration/foreign observation forbidden before tensor loading')
        path=Path(entry['directory'])/entry['filename'];identity=(str(path.resolve()),entry['sha256'])
        if identity in seen:raise ValueError('Duplicate source view')
        seen.add(identity)
        if manifest['files_sha256'].get(str(path.resolve()))!=entry['sha256']:raise ValueError('Source manifest/role hash mismatch')
        kind={'Car':0,'Building':1}.get(objects[entry['map_name']][str(entry['object_id'])]['object_type'],-1)
        if kind!=selected['target_kind']:raise ValueError('TRAIN target category changed')
        raw=read_training_source(entry,'train',allowed,selected['source_signature'])
        row=build_training_row(raw,target_kind=kind,real=family!='teacher_snapshot')
        positive=row['labels']['localization_utility']&row['labels']['utility_valid'];negative=~positive&row['labels']['utility_valid']
        if not positive.any() or not negative.any():raise ValueError('Each diagnostic view needs positive and hard negative candidates')
        rows.append(row);counts[family]=counts.get(family,0)+1
    if counts!={family:2 for family in families}:raise ValueError('Two TRAIN views from every original family required')
    if sha(PARENT)!=manifest['parent_sha256']:raise ValueError('Original warmstart changed')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=initial_model();initial=fingerprint(model.state_dict())
    if initial!=manifest['initial_weights_sha256']:raise ValueError('Original seed0 initial weights differ')
    require_idle_gpu();model=model.cuda().train();batch=to_device(collate_views(rows),'cuda');optimizer=optimizer_for(model)
    history=[];grads={}
    for step in range(6):
        torch.cuda.synchronize();tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        loss,stats=relation_v3_loss(model,model(**batch['network']),batch,refine=step>=3)
        if not torch.isfinite(loss):raise ValueError('Nonfinite CUDA loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,model.parameters()),5.,error_if_nonfinite=True)
        if not norm>0:raise ValueError('Missing actual CUDA gradient')
        grads={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
        if step>=3:
            for prefix in ('base.language.','base.scorer.rank_head.','base.match.','base.relation.','category.','risk.'):
                if not any(value>0 for name,value in grads.items() if name.startswith(prefix)):raise ValueError('Missing gradient '+prefix)
        optimizer.step();torch.cuda.synchronize();history.append(dict(step=step,refine=step>=3,loss=float(loss),seconds=time.perf_counter()-tick,stats=stats))
    if fingerprint(model.state_dict())==initial or sha(selection)!=expected or sha(roles_path)!=manifest['roles_sha256'] or code_hashes()!=manifest['code_sha256']:raise ValueError('No update or diagnostic inputs changed')
    report=dict(status='provisional_six_TRAIN_views_A2_CUDA_smoke_passed',formal_training=False,formal_gate=False,no_benchmark_used=True,
        selection_sha256=expected,script_sha256=sha(__file__),code_sha256=manifest['code_sha256'],roles_sha256=manifest['roles_sha256'],initial_weights_sha256=initial,
        source_counts=counts,smoke_steps=6,batch_views=6,device=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),history=history,gradients=grads,
        saved_or_reused_trained_weights=False,note='Ephemeral smoke optimizer updates are discarded. Official data audit, CPU/CUDA/runtime proofs and complete20+5 remain required after the full archive arrives.')
    save(report,output);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selection-sha256',required=True);args=parser.parse_args()
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args.selection_sha256)
