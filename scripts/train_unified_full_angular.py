"""Joint Geometry + Relation + optional DIRECT landmark bearing consistency TRAIN.

Use static-only, revised TRAIN JSONL produced by scripts/prepare_citynav.py.
Unmatched names/indirect instructions are NOT filtered. Frozen pretrained BERT
is an upstream cached feature source; no unobserved RGB or GT enters forward.
The existing PGVB visual weights remain available in the final integrated
checkpoint but are NOT re-trained by this static-only loss.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from graphnav.candidate_geometry import collate_full
from graphnav.unified_direct_bearing_loss import angular_consistency_loss
from graphnav.unified_polygon_group import PublicPolygonGroupDataset,grouped_cardinal_loss,audit_group_direction_vs_train_gt
from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.full_relation_training import (
    FORMAT, full_relation_inputs, load_full_relation_model,
)
from graphnav.static_localization import cross_entropy_goal
from graphnav.token_language import FrozenBERTTokens


def full_loader(path, args, *, shuffle):
    dataset = PublicPolygonGroupDataset(
        path, public_objects=args.public_objects,
        enable_predicates=args.predicate_mode=='group',grid_size=28,
        max_landmarks=args.max_landmarks,
        max_roads=args.max_roads,
        max_relations=args.max_relations,
        prior_sigma_m=args.prior_sigma_m,
        rule_sigma_m=args.rule_sigma_m,
        deduplicate_queries=False,
    )
    versions = {q.get("dataset_version") for q in dataset.records}
    matches = {q.get("match_mode") for q in dataset.records}
    if versions != {"revised"} or matches != {"fuzzy"}:
        raise ValueError(f"Full training requires revised/fuzzy, not {versions}/{matches}")
    # Source with step-sampled duplicates would not be a query-complete
    # experiment: require one original instruction/target record per row.
    if any("teacher_future" in q or "uav_pose" in q for q in dataset.records):
        raise ValueError("Prepare the source with --static-only, not teacher trajectory prefixes")
    return dataset, DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=0, collate_fn=collate_full,
    )


def paired_metrics(original, refined, batch):
    target = batch["target_xy"].to(refined.device)
    sizes = batch["map_sizes"].to(refined.device)
    grid = batch["grid_xy"].to(refined.device)
    indices = torch.stack((original.argmax(-1), refined.argmax(-1)), -1)
    predictions = grid.gather(1, indices[..., None].expand(-1, -1, 2))
    dist = torch.linalg.vector_norm(
        (predictions - target[:, None]) * sizes[:, None], dim=-1
    )
    return dist.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    for name in ("train-jsonl", "stage1", "original-v2", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-geometry", type=float, default=1e-5)
    p.add_argument("--lr-relation", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--public-objects", type=Path, required=True,
                   help="Full PUBLIC CityRefer objects.json for disambiguation, no GT")
    p.add_argument('--angular-loss-mode',choices=('none','unique_gt','group_word'),default='group_word',
                   help='none/control, original single-Polygon GT yaw, or new same-name Polygon OR relation')
    p.add_argument('--predicate-mode',choices=('off','group'),default='group',
                   help='Matched-public full-text control if off; multi-Polygon predicate signal if group')
    p.add_argument("--angular-weight", type=float, default=0.1,
                   help="0 enables predicate-only control; 0.1/0.3 angle ablations")
    p.add_argument("--kl-weight", type=float, default=0.05)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--min-train-records", type=int, default=20000,
                   help="Reject accidentally selected partial TRAIN JSONL")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Smoke-only; any value > 0 invalidates full training claim")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This full-scale training protocol requires CUDA")
    if a.epochs < 1 or a.batch_size < 1 or a.max_batches < 0 or a.angular_weight < 0 or a.kl_weight < 0:
        raise ValueError('Invalid full training config')
    if (a.angular_loss_mode=='none')!=(a.angular_weight==0) or (a.angular_weight>0 and a.predicate_mode=='off'):
        raise ValueError('Angular loss needs positive weight + group predicate signal')
    torch.set_num_threads(4)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    a.output.mkdir(parents=True, exist_ok=True)
    source_sha = dict(
        stage1=file_sha256(a.stage1),
        original_v2=file_sha256(a.original_v2),
        train_jsonl=file_sha256(a.train_jsonl),
        public_objects=file_sha256(a.public_objects),
    )
    original = torch.load(a.original_v2, map_location="cpu", weights_only=True)
    if original.get("format") != "unified-pgvb-v2":
        raise ValueError("Requires official integrated UnifiedPGVB-v2")
    if original["config"]["source"]["stage1_sha256"] != source_sha["stage1"]:
        raise ValueError("Original geometry/controller checksum mismatch")
    geometry_args = original["config"]["geometry_args"]
    if int(geometry_args["grid_size"]) != 28:
        raise ValueError("Unexpected controller input grid")
    # Rebuild the dataset using actual Geometry checkpoint feature protocol.
    actual = argparse.Namespace(
        batch_size=a.batch_size,
        max_landmarks=int(geometry_args["max_landmarks"]),
        max_roads=int(geometry_args["max_roads"]),
        max_relations=int(geometry_args["max_relations"]),
        prior_sigma_m=float(geometry_args["prior_sigma_m"]),
        rule_sigma_m=float(geometry_args["rule_sigma_m"]),
        public_objects=a.public_objects,
        predicate_mode=a.predicate_mode,
    )
    dataset, loader = full_loader(a.train_jsonl, actual, shuffle=True)
    if len(dataset) < 1:
        raise ValueError("Empty full CityNav TRAIN dataset")
    # Verify converter provenance: no silent strict-relation eligibility
    # filtering, and explicitly surface any malformed upstream episodes.
    provenance_path = Path(str(a.train_jsonl) + ".provenance.json")
    converter = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if converter is None or converter.get('binding_algorithm') != 'instruction_public_blind_PR23':
        raise ValueError('Use PR23 instruction-only PUBLIC-BLIND TRAIN: audit_unified_public_binding --emit-blind-train')
    if converter.get('public_catalog_sha256')!=source_sha['public_objects']:
        raise ValueError('TRAIN bind catalog does not equal current PUBLIC catalog SHA')
    if converter is not None:
        if (converter.get("output_sha256") != source_sha["train_jsonl"]
                or converter.get("static_only") is not True
                or converter.get("dataset_version_declared") != "revised"):
            raise ValueError("TRAIN converter provenance does not match source")
        if converter.get("records_generated") != len(dataset.records):
            raise ValueError("TRAIN records changed after conversion")
    # Every successfully converted source row stays in the denominator; no
    # direct-relation parser or GT-guided source selection.
    count = len(dataset.records)
    if not a.max_batches and count < a.min_train_records:
        raise ValueError(f"TRAIN source has only {count} records; expected full CityNav TRAIN")
    missing_anchor = sum(not row.get("landmarks") for row in dataset.records)
    manifest = dict(source_sha256=source_sha, records=count,
        experiment='unified-v2-public-polygon-set-v1',
        eligible_direct_predicate_audit=dict(dataset.binding_audit),
        public_binding_algorithm=converter['binding_algorithm'],
        angular_loss_mode=a.angular_loss_mode,
        predicate_mode=a.predicate_mode,
        set_loss_module_sha256=file_sha256('graphnav/unified_polygon_group.py'),
        angle_supervision_weight=a.angular_weight,
        kl_weight=a.kl_weight,
        angular_module_sha256=file_sha256("graphnav/unified_direct_bearing_loss.py"),
        explicit_predicates=a.predicate_mode=='group',
        GT_only_in_training_loss=True,
        geographic_north_CRS_unverified=True,
        no_public_anchor_rows=missing_anchor, source="complete revised CityNav TRAIN",
        converter_skipped_episodes=converter.get("episodes_skipped") if converter else None,
        converter_expected_records=converter.get("records_generated") if converter else None,
        eligibility_filter=None, checkpoint_format=FORMAT,
        epochs=a.epochs, batch_size=a.batch_size, seed=a.seed,
        min_train_records=a.min_train_records,
        smoke=bool(a.max_batches), max_batches=a.max_batches,
        lr_geometry=a.lr_geometry, lr_relation=a.lr_relation,
        training_code_sha256=file_sha256(__file__),
        relation_code_sha256=file_sha256('graphnav/aerialvg_inspired_relation.py'))
    manifest_path = a.output/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Resume inputs/configuration changed")
    if a.resume and not manifest_path.exists():
        raise ValueError("Resume requires existing training manifest")
    if not a.resume and (a.output/"last.pt").exists():
        raise FileExistsError("Training checkpoint exists; use --resume")
    atomic_json(manifest, manifest_path)

    previous = None
    if a.resume:
        previous = torch.load(a.output/"last.pt",
                              map_location="cpu", weights_only=True)
        if previous["source_sha256"] != source_sha:
            raise ValueError("Resume source fingerprint mismatch")
        model = load_full_relation_model(previous, "cuda")
        start_epoch = int(previous["completed_epochs"])
    else:
        model = load_full_relation_model(original, "cuda")
        start_epoch = 0
    model.requires_grad_(False)
    model.geometry.requires_grad_(True)
    model.full_relation.requires_grad_(True)
    # PGVB has no supervision from a static-only sample; keep it at verified
    # pretrained weights rather than silently training against fake pixels.
    optimizer = torch.optim.AdamW([
        dict(params=list(model.geometry.parameters()), lr=a.lr_geometry),
        dict(params=list(model.full_relation.parameters()), lr=a.lr_relation),
    ], weight_decay=0.001)
    if previous is not None:
        optimizer.load_state_dict(previous["optimizer"])
    bert = FrozenBERTTokens(
        geometry_args["text_backbone"], device="cuda",
        max_length=int(geometry_args["max_text_length"]),
    )
    bert_path = a.output/'bert_cache.pt'
    if bert_path.exists():bert.load_cache(bert_path)
    if previous is not None:
        torch.set_rng_state(previous["torch_rng"])
        torch.cuda.set_rng_state_all(previous["cuda_rng"])
        random.setstate(previous["python_rng"])
        n = previous["numpy_rng"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32),
                             n[2], n[3], n[4]))
    history = previous["history"] if previous is not None else []
    started = time.perf_counter()
    prior_seconds = float(previous.get('elapsed_seconds', 0)) if previous else 0.
    updates = int(previous.get('updates', 0)) if previous else 0
    torch.cuda.reset_peak_memory_stats()
    def save(completed, offset, seen, loss_total, corrected, damaged):
        ckpt = dict(format=FORMAT, model=model.state_dict(), config=original['config'],
            source_sha256=source_sha, completed_epochs=completed, planned_epochs=a.epochs,
            epoch_offset=offset, epoch_seen=seen, epoch_loss_total=loss_total,
            epoch_corrected=corrected, epoch_damaged=damaged, updates=updates,
            elapsed_seconds=prior_seconds+time.perf_counter()-started,
            optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),
            numpy_rng=(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]),
            history=history,smoke=bool(a.max_batches),
            angular_experiment=dict(weight=a.angular_weight,kl_weight=a.kl_weight,
                explicit_predicates=a.predicate_mode=='group',
                angular_loss_mode=a.angular_loss_mode,
                predicate_mode=a.predicate_mode,
                public_binding_algorithm='instruction_public_blind_PR23',
                protocol='same-name-public-polygon-group-v1'))
        atomic_save(ckpt,a.output/'last.pt')
    for epoch in range(start_epoch, a.epochs):
        model.train()
        continuing = previous is not None and epoch == start_epoch
        offset = int(previous.get('epoch_offset',0)) if continuing else 0
        loss_total = float(previous.get('epoch_loss_total',0)) if continuing else 0.
        ang_count=0
        angle_total=0.
        word_angle_matches=0
        literal_group_GT_matched=0
        literal_group_GT_total=0
        angle_gradient_checked=False
        seen = int(previous.get('epoch_seen',0)) if continuing else 0
        corrected = int(previous.get('epoch_corrected',0)) if continuing else 0
        damaged = int(previous.get('epoch_damaged',0)) if continuing else 0
        order = np.random.default_rng(a.seed+epoch).permutation(count).tolist()[offset:]
        loader = DataLoader(dataset,batch_size=a.batch_size,sampler=order,num_workers=0,
                            collate_fn=collate_full,generator=torch.Generator().manual_seed(a.seed+epoch))
        for index, batch in enumerate(loader):
            if a.max_batches and index >= a.max_batches:
                break
            torch.cuda.synchronize(); step_started=time.perf_counter()
            inputs = full_relation_inputs(batch, bert, "cuda")
            with torch.autocast("cuda", enabled=False):
                result = model.forward_full_relation(**inputs)
                base = result["geometry_field"].belief_logits.float()
                logits = result["field"].belief_logits.float()
                location = cross_entropy_goal(logits, batch["target_xy"].cuda())
                preserve = F.kl_div(
                    F.log_softmax(logits, -1),
                    F.softmax(base.detach(), -1), reduction="batchmean")
                if a.angular_loss_mode=='group_word':
                    angle=grouped_cardinal_loss(
                        logits,inputs['static_inputs']['pair_geometry'],
                        batch['angular_group_mask'].cuda(),
                        batch['anchor_predicate_ids'].cuda())
                elif a.angular_loss_mode=='unique_gt':
                    angle=angular_consistency_loss(
                        logits,inputs['static_inputs']['pair_geometry'],
                        batch['target_xy'].cuda(),inputs['static_inputs']['nodes'],
                        batch['map_sizes'].cuda(),batch['angular_anchor_index'].cuda())
                else:
                    angle=dict(loss=logits.sum()*0.,eligible_count=0,
                               angular_error_rad=logits.detach().sum()*0.,
                               valid=torch.zeros(len(logits),device=logits.device,dtype=torch.bool))
                loss = location + a.kl_weight * preserve + a.angular_weight * angle["loss"]
            optimizer.zero_grad(set_to_none=True)
            if a.angular_weight>0 and angle["eligible_count"] and not angle_gradient_checked:
                angular_logit_gradient=torch.autograd.grad(
                    angle["loss"],logits,retain_graph=True)[0]
                norm_angular=float(angular_logit_gradient.abs().sum().detach())
                if not np.isfinite(norm_angular) or norm_angular<=0:
                    raise RuntimeError("Angular consistency does not backpropagate into main Belief")
                angle_gradient_checked=True
                atomic_json({"angular_logit_gradient_L1":norm_angular,
                             "first_nonzero_update":updates},
                            a.output/"angular_gradient_audit.json")
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full TRAIN loss')
            loss.backward()
            if updates < 3:
                audit={name:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                       for name,module in [('Geometry',model.geometry),('Relation',model.full_relation)]}
                if not all(np.isfinite(v) and v>0 for v in audit.values()):raise RuntimeError('Missing/nonfinite core gradients')
                atomic_json(audit,a.output/'gradient_audit.json')
            norm=torch.nn.utils.clip_grad_norm_(
                list(model.geometry.parameters()) +
                list(model.full_relation.parameters()), 5.0,error_if_nonfinite=True)
            optimizer.step()
            size = len(batch["text"])
            seen += size
            loss_total += float(loss.detach()) * size
            ang_count += angle["eligible_count"]
            angle_total += float(angle["loss"].detach()) * angle["eligible_count"]
            if a.predicate_mode=="group":
                evidence=audit_group_direction_vs_train_gt(
                    batch["target_xy"].cuda(),inputs["static_inputs"]["nodes"],
                    batch["map_sizes"].cuda(),batch["angular_group_mask"].cuda(),
                    batch["anchor_predicate_ids"].cuda())
                literal_group_GT_matched+=evidence["matched_count"]
                literal_group_GT_total+=evidence["eligible_count"]
            if angle["eligible_count"] and a.angular_loss_mode=='unique_gt':
                eligible=angle["valid"]
                # Which fraction of explicit word directions matches observed
                # GT in map +X/CCW axes? This is an audit, NOT eligibility.
                # 1 east -> bin0, 2 west -> bin4, 3 north -> bin2, 4 south -> bin6.
                lookup=torch.tensor([0,0,4,2,6],device=logits.device)
                word_ids=batch["anchor_predicate_ids"].to(logits.device)[:,0].long()
                word_angle_matches+=int((angle["gt_angle_bin"][eligible]
                    == lookup[word_ids[eligible]]).sum().item())
            distances = paired_metrics(base.detach(), logits.detach(), batch)
            corrected += int(((distances[:, 0] > 20) &
                              (distances[:, 1] <= 20)).sum())
            damaged += int(((distances[:, 0] <= 20) &
                            (distances[:, 1] > 20)).sum())
            updates+=1;offset+=size;torch.cuda.synchronize()
            step=dict(update=updates,epoch=epoch+1,seen=seen,loss=float(loss.detach()),
                wall_ms=(time.perf_counter()-step_started)*1000,gradient_norm=float(norm),
                peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,
                relation_delta_max=float(result['relation_delta'].detach().abs().max()),
                angular_eligible=angle['eligible_count'],
                angular_loss=float(angle['loss'].detach()),
                angular_error_rad=float(angle['angular_error_rad']) if a.angular_loss_mode=='unique_gt' else None,
                angular_supported_mass_mean=(float(angle['supported_mass'].mean()) if a.angular_loss_mode=='group_word' else None))
            with (a.output/'steps.jsonl').open('a') as stream:stream.write(json.dumps(step)+'\n')
            if updates%200==0:
                save(epoch,offset,seen,loss_total,corrected,damaged)
            if index % 100 == 0:
                print(json.dumps(dict(epoch=epoch+1,batch=index,seen=seen,
                                      expected=count,loss=float(loss.detach()))),
                      flush=True)
        if not a.max_batches and seen != count:
            raise RuntimeError(f"Dataset coverage incomplete: {seen}/{count}")
        if a.angular_weight>0 and ang_count==0:
            raise RuntimeError("No eligible high-confidence direct named-world-cardinal task; audit parser/source before training")
        row = dict(epoch=epoch+1, observed=seen, expected=count,
                   angular_supervised=ang_count,
                   mean_angular_loss=angle_total/max(1,ang_count),
                   literal_group_GT_any_polygon_agreement=(
                       literal_group_GT_matched/max(1,literal_group_GT_total)
                       if literal_group_GT_total else None),
                   literal_group_GT_audited=literal_group_GT_total,
                   literal_8bin_GT_word_agreement=(word_angle_matches/max(1,ang_count)) if a.angular_loss_mode=='unique_gt' else None,
                   literal_8bin_agreement_N=ang_count,
                   angular_backprop_verified=bool(angle_gradient_checked) if a.angular_weight>0 else None,
                   train_loss=loss_total/max(seen,1),
                   train_fixes=corrected, train_breaks=damaged,
                   wall_seconds=time.perf_counter()-started)
        history.append(row)
        save(epoch+1,0,0,0.,0,0)
        bert.save_cache(bert_path)
        atomic_json(dict(**manifest, completed_epochs=epoch+1,
                         history=history, complete=(
                            epoch+1 == a.epochs and not a.max_batches)),
                    a.output/"progress.json")
        print(json.dumps(row), flush=True)
    if a.max_batches:
        print("WARNING: Smoke checkpoint is NOT a complete full-train result",flush=True)


if __name__ == "__main__":
    main()
