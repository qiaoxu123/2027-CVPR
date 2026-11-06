"""Matched direction-only experiment: text/name vs text/name + public Polygon neighborhood.

No navigation, no RGB, no frame rules, no GT angle at model forward.
Train and evaluate only on TRAIN with held-out maps. Both arms share the
same frozen BERT, target-angle labels, model capacity, optimizer and split.

Example:
python -m scripts.train_polygon_neighborhood_bearing \
  --train-jsonl /path/to/train_revised.jsonl \
  --existing-full-checkpoint /path/to/full_relation/last.pt \
  --public-objects /path/to/cityrefer_original/objects.json \
  --output /path/to/runs/polygon_bearing_s42 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import math
import random
import hashlib
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.polygon_neighborhood_bearing import (
    NeighborhoodBearingHead, PublicPolygonNeighborhood,
)
from graphnav.token_language import FrozenBERTTokens
from scripts.train_unified_angle_only import build_data


def choose_maps(records, seed=42, explicit=None):
    maps = sorted({row["map_name"] for row in records})
    if len(maps) < 3:
        raise ValueError("Map-disjoint holdout needs at least three TRAIN maps")
    if explicit:
        held = sorted(set(explicit))
        if not set(held).issubset(maps) or len(held) >= len(maps):
            raise ValueError("Invalid heldout map set")
    else:
        perm = np.random.default_rng(seed).permutation(len(maps))
        held = sorted(maps[int(i)] for i in perm[:max(2, round(len(maps) * .2))])
    return [i for i, row in enumerate(records) if row["map_name"] not in held], [
        i for i, row in enumerate(records) if row["map_name"] in held
    ], held


class BearingExamples(Dataset):
    """Only public reference/neighbor geometry enters inputs; GT is a label."""

    def __init__(self, records, catalog, *, max_refs=16, neighbors=24, radius_m=150.):
        self.records = records
        self.public = PublicPolygonNeighborhood(
            catalog, max_refs=max_refs, max_neighbors=neighbors,
            radius_m=radius_m,
        )
        self.max_refs = max_refs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        references = row["landmarks"]
        geo = self.public.for_references(row["map_name"], references)
        bounds = np.asarray(row["world_bounds"], np.float64)
        span = bounds[2:] - bounds[:2]
        gt_xy = np.asarray(row["target_xy"], np.float64)
        if gt_xy.shape != (2,) or np.any(span <= 0):
            raise ValueError("Invalid label coordinates or map bounds")
        # GT appears ONLY in this label block, never in PublicPolygonNeighborhood.
        target_world = bounds[:2] + gt_xy * span
        diff = target_world[None, :] - geo["anchor_centers_world"]
        dist = np.linalg.norm(diff, axis=-1)
        theta = np.mod(np.arctan2(diff[:, 1], diff[:, 0]), 2 * np.pi)
        labels = np.rint(theta / (2 * np.pi) * 8).astype(np.int64) % 8
        usable = geo["landmark_valid"] & (dist > 1e-9)
        labels[~usable] = -100
        angle = theta.astype(np.float32)
        angle[~usable] = np.nan
        names = [(str(x.get("object_type", "")) + " " + str(x["name"])).strip()
                 for x in references]
        names += [""] * (self.max_refs - len(names))
        result = dict(index=index, map_name=row["map_name"], text=row["instruction"],
                    names=names, labels=labels, true_angle=angle,
                    n_refs=int(geo["landmark_valid"].sum()),
                    **{key: geo[key] for key in (
                        "anchor_geometry", "neighbor_geometry", "neighbor_types",
                        "neighbor_valid", "landmark_valid"
                    )})
        return result


def collate(rows):
    keys = ("anchor_geometry", "neighbor_geometry", "neighbor_types",
            "neighbor_valid", "landmark_valid", "labels", "true_angle")
    batch = {key: torch.from_numpy(np.stack([r[key] for r in rows]))
             for key in keys}
    batch.update(text=[r["text"] for r in rows],
                 names=[r["names"] for r in rows],
                 map_name=[r["map_name"] for r in rows],
                 source_index=[r["index"] for r in rows],
                 n_refs=torch.tensor([r["n_refs"] for r in rows], dtype=torch.long))
    return batch


def predict(head, batch, bert, device, *, geometry_enabled):
    """GT-free forward: receives only explicit public/linguistic input keys."""
    with torch.no_grad():
        tokens, mask = bert.encode_tokens(batch["text"])
        names = bert.encode([name for group in batch["names"] for name in group])
        names = names.reshape(len(batch["text"]), len(batch["names"][0]), -1)
    return head(
        tokens=tokens.detach().to(device), token_mask=mask.to(device),
        node_emb=names.detach().to(device),
        landmark_valid=batch["landmark_valid"].to(device),
        anchor_geometry=batch["anchor_geometry"].to(device),
        neighbor_geometry=batch["neighbor_geometry"].to(device),
        neighbor_types=batch["neighbor_types"].to(device),
        neighbor_valid=batch["neighbor_valid"].to(device),
        geometry_enabled=geometry_enabled,
    )


def angular_loss(logits, labels):
    valid = labels >= 0
    if not valid.any():
        return logits.sum() * 0., 0, 0
    # Equal query weight: multiple public anchors do not multiply query weight.
    ce = F.cross_entropy(logits.transpose(1, 2), labels.clamp_min(0),
                         reduction="none")
    per_query = (ce * valid).sum(1) / valid.sum(1).clamp_min(1)
    chosen = valid.any(1)
    return per_query[chosen].mean(), int(valid.sum()), int(chosen.sum())


@torch.no_grad()
def evaluate(head, dataset, indices, bert, device, *, geometry_enabled,
             batch_size=8, max_batches=0):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, collate_fn=collate, num_workers=0)
    head.eval()
    rows = []
    covered = total_queries = 0
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        logits = predict(head, batch, bert, device,
                         geometry_enabled=geometry_enabled)
        labels = batch["labels"].to(device)
        angle = batch["true_angle"].to(device)
        valid = labels >= 0
        probs = logits.softmax(-1)
        best = logits.argmax(-1)
        top2 = logits.topk(2, dim=-1).indices
        pred_rad = best.float() * (2 * math.pi / head.bins)
        error = torch.atan2(torch.sin(pred_rad - angle),
                            torch.cos(pred_rad - angle)).abs() * 180 / math.pi
        nll = -probs.gather(-1, labels.clamp_min(0)[..., None]).squeeze(-1).clamp_min(1e-8).log()
        for i in range(len(batch["text"])):
            total_queries += 1
            ids = valid[i].nonzero().flatten().tolist()
            if ids:
                covered += 1
            for j in ids:
                rows.append(dict(index=batch["source_index"][i],
                                 map=batch["map_name"][i],
                                 n_refs=int(batch["n_refs"][i]),
                                 reference_index=j, true_bin=int(labels[i, j]),
                                 predicted_bin=int(best[i, j]),
                                 top1=int(best[i, j] == labels[i, j]),
                                 top2=int((top2[i, j] == labels[i, j]).any()),
                                 error_deg=float(error[i, j]),
                                 nll=float(nll[i, j]), probabilities=probs[i,j].cpu().tolist()))
    def stats(values):
        if not values:
            return dict(n=0, top1=None, top2=None, circular_MAE_deg=None,
                        median_error_deg=None, NLL=None)
        return dict(n=len(values), top1=float(np.mean([r["top1"] for r in values])),
                    top2=float(np.mean([r["top2"] for r in values])),
                    circular_MAE_deg=float(np.mean([r["error_deg"] for r in values])),
                    median_error_deg=float(np.median([r["error_deg"] for r in values])),
                    NLL=float(np.mean([r["nll"] for r in values])))
    return dict(all=stats(rows),
                single_polygon=stats([r for r in rows if r["n_refs"] == 1]),
                multi_polygon=stats([r for r in rows if r["n_refs"] > 1]),
                supported_queries=covered, evaluated_queries=total_queries,
                by_map={m:stats([r for r in rows if r["map"] == m])
                        for m in sorted({r["map"] for r in rows})}), rows



def normalize_instruction(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def isolated_split(records, catalog, *, seed=42):
    fit,held,maps=choose_maps(records,seed=seed)
    # Map split fixed before reading labels. Purge shared text and identical
    # public target contours by identity metadata only, never by target distance.
    held_text={normalize_instruction(records[i]['instruction']) for i in held}
    def physical(row):
        parts=str(row.get('episode_id','')).split(':')
        oid=parts[1] if len(parts)>1 else ''
        obj=catalog.get(row['map_name'],{}).get(oid,{})
        xy=np.asarray(obj.get('contour',[]),dtype=np.float64)
        if xy.ndim==2 and xy.shape[-1]==2 and len(xy):
            # sorted world vertices avoid different cyclic starts/orderings.
            vertices=sorted(map(tuple,np.round(xy,5).tolist()))
            return hashlib.sha256(json.dumps(vertices).encode()).hexdigest()
        return row['map_name']+':'+oid
    held_phys={physical(records[i]) for i in held};purged=[];clean=[]
    for i in fit:
        reasons=[]
        if normalize_instruction(records[i]['instruction']) in held_text:reasons.append('duplicate_instruction')
        if physical(records[i]) in held_phys:reasons.append('same_public_target_contour')
        (purged if reasons else clean).append(dict(index=i,reasons=reasons) if reasons else i)
    assert not {records[i]['map_name'] for i in clean}&set(maps)
    assert not {normalize_instruction(records[i]['instruction']) for i in clean}&held_text
    assert not {physical(records[i]) for i in clean}&held_phys
    return clean,held,dict(heldout_maps=maps,fit_maps=sorted({records[i]['map_name'] for i in clean}),purged=purged,fit_queries=len(clean),heldout_queries=len(held),physical_overlap=0,normalized_instruction_overlap=0)


def state_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('train-jsonl','existing-full-checkpoint','public-objects','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=1e-4);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--expected-count',type=int,default=21878);parser.add_argument('--radius-m',type=float,default=150.)
    parser.add_argument('--neighbors',type=int,default=24);parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--max-train-batches',type=int,default=0);parser.add_argument('--max-eval-batches',type=int,default=0)
    parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    if a.seed!=42 or a.epochs<1 or a.batch_size<1:raise ValueError('Invalid fixed training settings')
    if a.output.exists() and not a.resume:raise FileExistsError('Preserve previous experiments')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    a.output.mkdir(parents=True,exist_ok=True);started=time.perf_counter();torch.set_num_threads(4)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;device=torch.device(a.device)
    source=torch.load(a.existing_full_checkpoint,weights_only=True,map_location='cpu')
    ds=build_data(a.train_jsonl,source,expected=a.expected_count,public_objects=a.public_objects)
    catalog=json.loads(a.public_objects.read_text());cfg=source['config']['geometry_args']
    dataset=BearingExamples(ds.records,catalog,max_refs=int(cfg['max_landmarks']),neighbors=a.neighbors,radius_m=a.radius_m)
    train_ids,val_ids,split=isolated_split(ds.records,catalog)
    bert=FrozenBERTTokens(cfg['text_backbone'],device=device,max_length=512)
    # Safe reuse of untruncated entries only. Never silently use a96-token
    # cache item for a longer original instruction.
    oldpath=a.existing_full_checkpoint.parent/'bert_cache.pt';reused=0
    if oldpath.exists():
        old=torch.load(oldpath,weights_only=True,map_location='cpu')
        if (old['name'],old['revision'])!=(bert.name,bert.revision):raise ValueError('BERT cache/model mismatch')
        entries=list(old['cache']);lengths=[len(x) for x in bert.tokenizer(entries,add_special_tokens=True,truncation=False)['input_ids']]
        for text,n in zip(entries,lengths):
            if n<=old['max_length']:bert.cache[text]=old['cache'][text];reused+=1
        del old
    texts=list(dict.fromkeys([r['instruction'] for r in ds.records]+[(r.get('object_type','')+' '+r['name']).strip() for q in ds.records for r in q['landmarks']]+['']))
    lens=[len(x) for x in bert.tokenizer(texts,add_special_tokens=True,truncation=False)['input_ids']]
    if max(lens)>512:raise ValueError('Full instruction exceeds512: implement explicit windowing, never truncate silently')
    print(json.dumps(dict(stage='bert_cache',texts=len(texts),safe_reused=reused,max_tokens=max(lens))),flush=True)
    for j in range(0,len(texts),64):bert._load(texts[j:j+64])
    bert.save_cache(a.output/'bert_cache.pt')
    inventory=Counter();neighbor_counts=[]
    for i in range(len(dataset)):
        q=dataset[i];n=int((q['labels']>=0).sum());inventory['queries']+=1;inventory['valid_anchors']+=n
        inventory['queries_with_labels' if n else 'no_usable_angle_queries']+=1
        inventory['single_queries' if q['n_refs']==1 else 'multi_queries' if q['n_refs']>1 else 'no_public_anchor_queries']+=1
        neighbor_counts.extend(q['neighbor_valid'][q['landmark_valid']].sum(-1).tolist())
    config=dict(format='polygon-neighborhood-bearing-v2',seed=42,epochs=a.epochs,batch=a.batch_size,lr=a.lr,weight_decay=.01,clip=1.,radius_m=a.radius_m,neighbors=a.neighbors,n_train=len(train_ids),n_holdout=len(val_ids),expected_count=a.expected_count,max_train_batches=a.max_train_batches,max_eval_batches=a.max_eval_batches,train_sha256=file_sha256(a.train_jsonl),catalog_sha256=file_sha256(a.public_objects),source_sha256=file_sha256(a.existing_full_checkpoint),model_code_sha256=file_sha256(Path('graphnav/polygon_neighborhood_bearing.py')),training_code_sha256=file_sha256(__file__),BERT_name=bert.name,BERT_revision=bert.revision,BERT_max_tokens=512,actual_max_tokens=max(lens),cache_safe_reused=reused,split=split,inventory=dict(inventory),neighbor_count_mean=float(np.mean(neighbor_counts)),no_RGB_no_navigation_no_frame_rules=True,fixed_final_epoch_not_selected_on_holdout=True)
    configpath=a.output/'protocol.json'
    if configpath.exists():assert json.loads(configpath.read_text())==config,'Resume inputs changed'
    else:atomic_json(config,configpath)
    initial=None;results={}
    for name,use_geometry in [('text_name_control',False),('public_polygon_neighborhood',True)]:
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        head=NeighborhoodBearingHead().to(device);digest=state_digest(head.state_dict())
        if initial is None:initial=digest
        else:assert digest==initial,'Different A/B initialization'
        optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=.01)
        history=[];start_epoch=0;path=a.output/(name+'_last.pt')
        if a.resume and path.exists():
            state=torch.load(path,map_location=device,weights_only=True);assert state['protocol']==config
            head.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];start_epoch=state['epoch']
            torch.set_rng_state(state['rng_cpu'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['rng_cuda']])
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start_epoch,a.epochs):
            order=np.random.default_rng(42+epoch).permutation(train_ids).tolist()
            loader=DataLoader(Subset(dataset,order),batch_size=a.batch_size,collate_fn=collate,num_workers=0)
            head.train();loss_total=labeled_total=queries_total=batches=0;tick=time.perf_counter()
            for step,batch in enumerate(loader):
                if a.max_train_batches and step>=a.max_train_batches:break
                logits=predict(head,batch,bert,device,geometry_enabled=use_geometry)
                loss,labeled,queries=angular_loss(logits,batch['labels'].to(device))
                if not labeled:continue
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if batches==0:
                    grad={k:float(p.grad.abs().sum()) for k,p in head.named_parameters() if p.grad is not None}
                    assert sum(grad.values())>0 and all(np.isfinite(v) for v in grad.values())
                    assert all(p.grad is None and not p.requires_grad for p in bert.model.parameters())
                    if use_geometry:assert grad.get('neighbor.weight',0)>0 and grad.get('anchor.weight',0)>0
                    atomic_json(dict(epoch=epoch+1,gradient_L1=grad,BERT_frozen=True,initial_digest=digest,parameters=sum(p.numel() for p in head.parameters())),a.output/(name+'_gradient.json'))
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                loss_total+=float(loss.detach())*queries;labeled_total+=labeled;queries_total+=queries;batches+=1
                if step%100==0:print(json.dumps(dict(arm=name,epoch=epoch+1,step=step,queries=queries_total,loss=float(loss))),flush=True)
            if not batches:raise RuntimeError('No angle labels')
            torch.cuda.synchronize();entry=dict(epoch=epoch+1,mean_loss=loss_total/queries_total,supervised_anchors=labeled_total,supervised_queries=queries_total,updates=batches,seconds=time.perf_counter()-tick,peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30)
            history.append(entry)
            atomic_save(dict(format=config['format'],arm=name,model=head.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,protocol=config,history=history,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),initial_digest=digest,trained_on_heldout=False),path)
            atomic_json(history,a.output/(name+'_history.json'));print(json.dumps(dict(arm=name,**entry)),flush=True)
        report,rows=evaluate(head,dataset,val_ids,bert,device,geometry_enabled=use_geometry,batch_size=a.batch_size,max_batches=a.max_eval_batches)
        results[name]=report
        with (a.output/(name+'_heldout_rows.jsonl')).open('w') as out:
            for item in rows:out.write(json.dumps(item)+'\n')
    atomic_json(dict(protocol=config,arms=results,NOT_a_navigation_result=True,supervised_direction_is_position_not_proven_language_semantics=True,seconds=time.perf_counter()-started),a.output/'REPORT.json')
    atomic_json(dict(records=ds.records,fit_indices=train_ids,heldout_indices=val_ids),a.output/'data_manifest.json')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
