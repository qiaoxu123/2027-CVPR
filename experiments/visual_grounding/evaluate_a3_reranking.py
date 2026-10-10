"""Frozen Epoch14 A3 static localization and causal offline visual evidence.

Human trajectory prefixes are offline evidence, not online policy rollouts.
Private target coordinates are used only for metrics after the public forward.
Unknown dense cells retain their exact static score during reranking.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.full_experiment import new_model, _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.checkpointing import file_sha256, provenance
from graphnav.semantic_geometry import point_inside_polygon

PINNED_A3_SHA = '9d9be861bf7e9cac986d747867990de3d409ccb9bc1545862d71a620448380d7'


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(s) for s in f if s.strip()]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + '\n')



def validate_aligned_query_rows(static_rows, evidence_rows, expected_count=None):
    """Reject missing, reordered or cross-episode rows before any zip iteration."""
    if len(static_rows) != len(evidence_rows):
        raise ValueError('Static/evidence row count mismatch; refusing silent truncation')
    if expected_count is not None and len(static_rows) != expected_count:
        raise ValueError('Static/evidence rows differ from dense query count')
    keys = ('query_index', 'map_name', 'source_episode_index')
    for i, (static_row, evidence_row) in enumerate(zip(static_rows, evidence_rows)):
        for key in keys:
            if key not in static_row or key not in evidence_row:
                raise ValueError(f'Missing alignment key {key} at row {i}')
            if static_row[key] != evidence_row[key]:
                raise ValueError(f'Static/evidence {key} mismatch at row {i}')
    ids = [row['query_index'] for row in static_rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate static/evidence query_index')
    if expected_count is not None and set(ids) != set(range(expected_count)):
        raise ValueError('Static/evidence query indices do not cover dense query rows')
    return len(ids)


def episode_lookup(path):
    lookup={}
    for i,row in enumerate(json.loads(Path(path).read_text())):
        name=f"{row['area']}_block_{row['block']}"
        for text in row['descriptions']:
            lookup.setdefault((name,text),i)
    return lookup


def static(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if (out/'static_dense.npz').exists():
        raise ValueError('Refusing to overwrite existing frozen static evaluation')
    sha = file_sha256(a.checkpoint)
    if sha != PINNED_A3_SHA:
        raise ValueError('Expected frozen A3 Seed1 Epoch14 checkpoint')
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    source_lookup=episode_lookup(a.raw_episodes)
    if state['epoch'] != 14:
        raise ValueError('Unexpected checkpoint epoch')
    cfg = SimpleNamespace(**state['args']); device = torch.device(a.device)
    model = new_model(cfg, device).eval().requires_grad_(False)
    model.load_state_dict(state['model'], strict=True)
    encoder = FrozenBERTTokens(cfg.text_backbone, device=a.device,
                               max_length=getattr(cfg, 'max_text_length', 96))
    # Preserve the training-name cache precedence of the audited matched-cache run.
    fingerprints = state['input_fingerprint']; caches = {}
    for key, explicit in [('train_bert_cache', a.train_bert_cache),
                          ('seen_bert_cache', a.seen_bert_cache)]:
        path = explicit or fingerprints[key]['path']
        if file_sha256(path) != fingerprints[key]['sha256']:
            raise ValueError(f'{key} fingerprint differs from training')
        caches[key] = {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)}
        encoder.load_cache(path)
        if key == 'train_bert_cache':
            train_names = encoder.cache.copy()
    encoder.cache.update(train_names)
    ds = FullCandidateDataset(a.records, grid_size=cfg.grid_size,
        max_landmarks=cfg.max_landmarks, max_roads=cfg.max_roads,
        max_relations=cfg.max_relations, prior_sigma_m=cfg.prior_sigma_m,
        rule_sigma_m=cfg.rule_sigma_m)
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers,
                        collate_fn=collate_full, pin_memory=device.type == 'cuda')
    parts = defaultdict(list); predictions = []; timer = time.perf_counter(); offset = 0
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader:
            inputs, tokens, mask, names = _batch_inputs(batch, encoder, device, 'geometry')
            if 'target_xy' in inputs:
                raise ValueError('Private evaluator target leaked into model inputs')
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                field = model(inputs, variant='geometry', tokens=tokens, token_mask=mask, node_emb=names)
            logits = field.belief_logits.float()
            if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite A3 logits')
            top = logits.topk(20, dim=-1).indices
            top8 = logits.topk(8, dim=-1).indices
            bounds = torch.tensor(batch['world_bounds'], device=device, dtype=torch.float32)
            xy = bounds[:, None, :2] + inputs['grid_xy'].float() * (bounds[:, None, 2:]-bounds[:, None, :2])
            # Access evaluator-only target after model inference, never in evidence selection.
            target = bounds[:, :2] + batch['target_xy'].to(device) * (bounds[:, 2:]-bounds[:, :2])
            top_xy = xy.gather(1, top[..., None].expand(-1, -1, 2))
            distance = (top_xy-target[:, None]).norm(dim=-1).cpu().numpy()
            top8_xy = xy.gather(1, top8[..., None].expand(-1, -1, 2))
            distance8 = (top8_xy-target[:, None]).norm(dim=-1).cpu().numpy()
            for i in range(len(batch['text'])):
                record = ds.records[offset+i]
                predictions.append(dict(query_index=offset+i, map_name=record['map_name'],
                    instruction=record['instruction'], episode_id=record.get('episode_id'),
                    source_episode_index=source_lookup[(record['map_name'],record['instruction'])],
                    top20_ids=top[i].cpu().tolist(), top20_world_xy=top_xy[i].cpu().tolist(),
                    top8_ids=top8[i].cpu().tolist(),top8_distance_m=distance8[i].tolist(),
                    distance_m=distance[i].tolist(), NE_m=float(distance[i,0]),
                    **{f'R@{k}/20m':bool((distance8[i] if k==8 else distance[i,:k]).min()<=20)
                       for k in (1,8,20)}))
            for key,value in [('logits',logits),('world_xy',xy),('target_world_xy',target)]:
                parts[key].append(value.cpu().numpy())
            offset += len(batch['text'])
            if offset % 256 == 0: print(json.dumps({'completed':offset,'total':len(ds)}),flush=True)
    if device.type == 'cuda': torch.cuda.synchronize()
    np.savez_compressed(out/'static_dense.npz', **{k:np.concatenate(v) for k,v in parts.items()})
    (out/'static_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    metrics = {f'R@{k}/20m':float(np.mean([r[f'R@{k}/20m'] for r in predictions])) for k in (1,8,20)}
    wrong = [r for r in predictions if not r['R@1/20m']]
    metrics.update(samples=len(predictions), NE_m=float(np.mean([r['NE_m'] for r in predictions])),
        top1_errors=len(wrong), **{f'wrong_top1_recoverable_top{k}_fraction':
        float(np.mean([r[f'R@{k}/20m'] for r in wrong])) for k in (8,20)})
    report = dict(checkpoint=str(Path(a.checkpoint).resolve()), checkpoint_sha256=sha, epoch=14,
        records_sha256=file_sha256(a.records), bert_caches=caches, model_load='strict all keys',
        precision='bf16 forward; float32 topk', selection='first map+instruction; label independent',
        metrics=metrics, seconds=time.perf_counter()-timer, provenance=provenance(),
        torch=torch.__version__, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30
        if device.type=='cuda' else None, note='Static 20m localization; not navigation success.')
    write_json(out/'static_metrics.json', report); print(json.dumps(report,indent=2),flush=True)


def evidence(a):
    out = Path(a.output); data = Path(a.manifest_dir)
    frozen_sha={name:file_sha256(data/name) for name in ('manifest.jsonl','regions.jsonl','objects.json')
                if (data/name).exists()}
    predictions = read_jsonl(out/'static_predictions.jsonl')
    source_lookup=episode_lookup(a.raw_episodes)
    for pred in predictions:
        pred['source_episode_index']=source_lookup[(pred['map_name'],pred['instruction'])]
    # This repairs metadata only; dense inference and evaluator labels are untouched.
    (out/'static_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    with np.load(out/'static_dense.npz') as archive:
        dense={k:archive[k] for k in archive.files}
    manifest = read_jsonl(data/'manifest.jsonl'); regions = read_jsonl(data/'regions.jsonl')
    objects = json.loads((data/'objects.json').read_text()) if (data/'objects.json').exists() else {}
    by_episode = defaultdict(dict)
    # Both manifests may contain identical sample IDs; count each real observation once.
    for row in regions+manifest:
        if row['observed_step'] > row['observation_cutoff_step']:
            raise ValueError('Future image in evidence manifest')
        by_episode[int(row['source_episode_index'])][row['sample_id']] = row
    rows = []; geometry_cache={}
    for pred in predictions:
        episode = list(by_episode[pred['source_episode_index']].values())
        if any(r['map_name'] != pred['map_name'] for r in episode):
            raise ValueError('Source episode/map mismatch')
        cutoff = max((r['observation_cutoff_step'] for r in episode), default=-1)
        slot_rows = defaultdict(list); valid = []
        for r in episode:
            if not r.get('roi_quality',{}).get('attribute_resolution_eligible',False): continue
            if not r.get('roi_path') or not Path(r['roi_path']).exists(): continue
            if r['observed_step'] > cutoff: raise ValueError('Future ROI after query cutoff')
            poly = r.get('object_geometry')
            if poly is None:
                obj = objects.get(r['map_name'],{}).get(str(r['object_id']),{})
                poly = obj.get('contour')
            poly = np.asarray(poly if poly is not None else [], dtype=np.float32)
            if poly.ndim!=2 or poly.shape[0]<3: continue
            grid=dense['world_xy'][pred['query_index']]
            geometry_key=(r['map_name'],str(r['object_id']),tuple(grid[0]),tuple(grid[-1]))
            if geometry_key not in geometry_cache:
                geometry_cache[geometry_key]=point_inside_polygon(grid,poly)
            inside=geometry_cache[geometry_key]
            for slot in np.flatnonzero(inside):
                slot_rows[int(slot)].append(dict(sample_id=r['sample_id'],object_id=str(r['object_id']),
                    object_type=r['object_type'], observed_step=r['observed_step'],
                    attributes=r.get('attributes',{}), roi_path=r['roi_path']))
            valid.append(r)
        target_visible = any(r.get('is_episode_target',False) for r in episode)
        target_usable = any(r.get('is_episode_target',False) for r in valid)
        top = pred['top20_ids']; success = pred['distance_m']
        ranks={1:(top[:1],success[:1]),8:(pred['top8_ids'],pred['top8_distance_m']),20:(top,success)}
        rows.append(dict(query_index=pred['query_index'],map_name=pred['map_name'],
            source_episode_index=pred['source_episode_index'],observation_cutoff_step=cutoff,
            evidence_source='fixed sparse human-prefix offline; not A3 policy rollout',
            target_visible_evaluator_only=target_visible,target_usable_evaluator_only=target_usable,
            observed_instances=len({r['object_id'] for r in valid}),
            known_dense_slots={str(k):v for k,v in slot_rows.items()},
            **{f'top{k}_has_known_slot':any(i in slot_rows for i in ranks[k][0]) for k in (1,8,20)},
            **{f'correct_top{k}_slot_known_evaluator_only':any(i in slot_rows and d<=20
                for i,d in zip(*ranks[k])) for k in (8,20)}))
        if len(rows)%256==0:print(json.dumps(dict(evidence_queries=len(rows),total=len(predictions))),flush=True)
    (out/'evidence_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    metric = dict(queries=len(rows), source_episodes_observed=len(by_episode),
        target_visible_fraction=float(np.mean([r['target_visible_evaluator_only'] for r in rows])),
        target_high_resolution_materialized_roi_fraction=float(np.mean([r['target_usable_evaluator_only'] for r in rows])),
        **{f'top{k}_known_slot_coverage':float(np.mean([r[f'top{k}_has_known_slot'] for r in rows])) for k in (1,8,20)},
        reliable_visual_attribute_labels=sum('verified' in v.get('confidence','') and 'not_' not in v.get('confidence','')
            for r in manifest+regions for v in r.get('attributes',{}).values()),
        limitations='Sparse human-prefix coverage lower bound; weak polygon projection; geometry overlap is not semantic verification.')
    for k in (8,20):
        eligible = [i for i,p in enumerate(predictions) if not p['R@1/20m'] and p[f'R@{k}/20m']]
        metric[f'recoverable_top{k}_correct_known_fraction'] = float(np.mean([
            rows[i][f'correct_top{k}_slot_known_evaluator_only'] for i in eligible])) if eligible else None
    if any(file_sha256(data/name)!=sha for name,sha in frozen_sha.items()):
        raise ValueError('Evidence source changed during evaluation; freeze full manifest first')
    write_json(out/'evidence_metrics.json',dict(metrics=metric,input_sha256=frozen_sha))
    print(json.dumps(metric,indent=2),flush=True)


def scores(a):
    """Use cached, protocol-pinned embeddings; never encode selected GT viewpoints."""
    out=Path(a.output); frozen=Path(a.frozen_index)
    state=torch.load(a.embeddings,map_location='cpu',weights_only=True)
    if state['protocol']['frozen_sha256']!=file_sha256(frozen/'frozen.json'):
        raise ValueError('Visual embeddings and frozen index differ')
    index=read_jsonl(frozen/'index.jsonl'); lookup={r['sample_id']:i for i,r in enumerate(index)}
    if len(index)!=len(state['images']):raise ValueError('Image/index count mismatch')
    text_lookup={s:i for i,s in enumerate(state['strings'])}
    phrases={}
    for q in sorted(read_jsonl(frozen/'queries.jsonl'),key=lambda q:q['attribute_phrase']):
        phrases.setdefault((q['map_name'],q['instruction']),q['attribute_phrase'])
    preds=read_jsonl(out/'static_predictions.jsonl'); ev=read_jsonl(out/'evidence_predictions.jsonl')
    validate_aligned_query_rows(preds, ev)
    rows=[]; covered_queries=0
    for p,e in zip(preds,ev):
        text=p['instruction'] if a.language_input=='instruction' else phrases.get((p['map_name'],p['instruction']))
        if text not in text_lookup:continue
        ti=text_lookup[text]; sample_ids=set()
        for views in e['known_dense_slots'].values():
            for view in views:
                sid=view['sample_id']
                if sid not in lookup:continue
                r=index[lookup[sid]]
                if int(r['source_episode_index'])!=p['source_episode_index']:
                    raise ValueError('Cross-episode visual evidence leak')
                if r['observed_step']>e['observation_cutoff_step']:
                    raise ValueError('Future visual evidence leak')
                sample_ids.add(sid)
        if sample_ids:covered_queries+=1
        for sid in sorted(sample_ids):
            value=float(state['images'][lookup[sid]].float() @ state['texts'][ti].float())
            if not np.isfinite(value):raise FloatingPointError('Nonfinite visual score')
            rows.append(dict(query_index=p['query_index'],sample_id=sid,score=value))
    tag=a.result_tag or state['protocol']['model']+'_'+a.language_input
    (out/(tag+'_scores.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(out/(tag+'_scores_provenance.json'),dict(embeddings_sha256=file_sha256(a.embeddings),
        frozen_sha256=file_sha256(frozen/'frozen.json'),language_input=a.language_input,
        model_protocol=state['protocol'],queries_with_scores=covered_queries,sample_query_scores=len(rows),
        phrase_selection='lexicographically first conservative audited phrase per map+instruction; fixed before scores',
        inference='normalized cosine on same-episode observed cached ROIs; no target-selected views'))
    print(json.dumps(dict(tag=tag,queries_with_scores=covered_queries,scores=len(rows))),flush=True)


def rerank(a):
    out = Path(a.output); pred = read_jsonl(out/'static_predictions.jsonl')
    evidence_rows = read_jsonl(out/'evidence_predictions.jsonl')
    with np.load(out/'static_dense.npz') as archive:
        dense={k:archive[k] for k in archive.files}
    validate_aligned_query_rows(pred, evidence_rows, expected_count=len(dense['logits']))
    if any(len(dense[key]) != len(pred) for key in ('world_xy', 'target_world_xy')):
        raise ValueError('Dense coordinates differ from static query count')
    scores = {(int(r['query_index']),r['sample_id']):float(r['score']) for r in read_jsonl(a.visual_scores)}
    results = []
    for p,e in zip(pred,evidence_rows):
        qi = p['query_index']; logits = dense['logits'][qi].copy(); known = {}; groups = defaultdict(list)
        slot_identity = {}
        for slot,views in e['known_dense_slots'].items():
            suitable = []
            for v in views:
                labels = list(v['attributes'].values())
                if a.evidence_tier=='visual_verified':
                    labels=[x for x in labels if 'verified' in x.get('confidence','') and
                            'not_' not in x.get('confidence','') and 'weak' not in x.get('confidence','')]
                if labels and (qi,v['sample_id']) in scores: suitable.append(v)
            # Overlapping object polygons cannot safely claim a unique semantic cell.
            ids = {v['object_id'] for v in suitable}; types={v['object_type'] for v in suitable}
            if len(ids)!=1 or len(types)!=1: continue
            slot=int(slot);known[slot]=max(scores[(qi,v['sample_id'])] for v in suitable)
            slot_identity[slot]=next(iter(ids))
            groups[next(iter(types))].append(slot)
        changed=False
        for slots in groups.values():
            if len({slot_identity[s] for s in slots})<2: continue
            ordered=sorted(slots,key=lambda s:(-known[s],-float(logits[s]),s))
            # Preserve the entire static-score multiset; Unknown remains unchanged.
            vals=sorted((float(logits[s]) for s in slots),reverse=True)
            for s,v in zip(ordered,vals):logits[s]=v
            changed=True
        unknown=np.ones(len(logits),dtype=bool);unknown[list(known)]=False
        assert np.array_equal(logits[unknown],dense['logits'][qi][unknown])
        # Preserve the audited CUDA ranking among unchanged tied maxima. A CPU
        # topk implementation can pick a different tied cell even at zero coverage.
        maximum=float(logits.max()); original_top=p['top20_ids'][0]
        if float(logits[original_top])==maximum:
            top=original_top
        else:
            prior_max=[s for s in p['top20_ids'] if float(logits[s])==maximum]
            top=prior_max[0] if prior_max else int(np.flatnonzero(logits==maximum)[0])
        dist=float(np.linalg.norm(dense['world_xy'][qi,top]-dense['target_world_xy'][qi]))
        results.append(dict(query_index=qi,map_name=p['map_name'],static_correct=p['R@1/20m'],
            reranked_correct=dist<=20,static_top1=p['top20_ids'][0],reranked_top1=top,
            known_scored_cells=len(known),available_permutation=changed,NE_m=dist))
    wrong=sum(not r['static_correct'] for r in results);correct=len(results)-wrong
    report=dict(evidence_tier=a.evidence_tier,visual_scores_sha256=file_sha256(a.visual_scores),
        samples=len(results),Static_R1=float(np.mean([r['static_correct'] for r in results])),
        Visual_Reranked_R1=float(np.mean([r['reranked_correct'] for r in results])),
        Recovery_Rate=sum(not r['static_correct'] and r['reranked_correct'] for r in results)/max(1,wrong),
        Damage_Rate=sum(r['static_correct'] and not r['reranked_correct'] for r in results)/max(1,correct),
        Available_Visual_Evidence_Coverage=float(np.mean([r['available_permutation'] for r in results])),
        unknown_scores_exactly_preserved=True,
        note='Weak-language tier is exploratory and not independently visually verified supervision.'
             if a.evidence_tier=='language_weak' else 'Trusted visual-attribute gate applied.')
    delta=np.asarray([int(r['reranked_correct'])-int(r['static_correct']) for r in results])
    rng=np.random.default_rng(0)
    iid=[float(delta[rng.integers(0,len(delta),len(delta))].mean()) for _ in range(2000)]
    maps=sorted({r['map_name'] for r in results})
    map_arrays={m:delta[[r['map_name']==m for r in results]] for m in maps}
    clustered=[]
    for _ in range(2000):
        drawn=rng.choice(maps,size=len(maps),replace=True)
        clustered.append(float(np.concatenate([map_arrays[m] for m in drawn]).mean()))
    report.update(paired_bootstrap_delta_R1_ci95=np.quantile(iid,[.025,.975]).tolist(),
        map_cluster_bootstrap_delta_R1_ci95=np.quantile(clustered,[.025,.975]).tolist(),
        bootstrap_replicates=2000,bootstrap_seed=0,independent_maps=len(maps),
        per_map={m:dict(samples=sum(r['map_name']==m for r in results),
            Static_R1=float(np.mean([r['static_correct'] for r in results if r['map_name']==m])),
            Visual_Reranked_R1=float(np.mean([r['reranked_correct'] for r in results if r['map_name']==m])))
            for m in maps})
    tag=a.result_tag or a.evidence_tier
    (out/(tag+'_predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in results))
    write_json(out/(tag+'_metrics.json'),report);print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['static','evidence','scores','rerank'])
    p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--records')
    p.add_argument('--raw-episodes',default='/home/rental/20260922_1/Workspace/DATA/sbfnav-data/processed_citynav/citynav_val_seen.json')
    p.add_argument('--train-bert-cache');p.add_argument('--seen-bert-cache');p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--manifest-dir');p.add_argument('--visual-scores');p.add_argument('--result-tag')
    p.add_argument('--embeddings');p.add_argument('--frozen-index')
    p.add_argument('--language-input',choices=['instruction','attribute_phrase'],default='instruction')
    p.add_argument('--evidence-tier',choices=['visual_verified','language_weak'],default='visual_verified')
    a=p.parse_args();{'static':static,'evidence':evidence,'scores':scores,'rerank':rerank}[a.stage](a)


if __name__=='__main__': main()
