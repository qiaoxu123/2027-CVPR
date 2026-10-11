"""Read-only previous resources; label-free caches and labels kept separate."""
from pathlib import Path
import json,numpy as np,torch
from experiments.semantic_polygon_grounding.audit import sha
from experiments.language_object_graph.objects import ObjectGraph

VERIFIED=Path('/home/rental/20260922_1/Workspace/Ablations/GraphNav/rgb-language-grounding/runs/supervised_object_grounding_verified_a3')
OUT=Path('runs/p6_latent_object_binding')

def baseline():
    proof=json.loads((VERIFIED/'a3_complete.json').read_text())
    if not proof.get('loaded_state_equals_checkpoint') or sha(VERIFIED/'a3.pt')!=proof['cache_sha']:raise ValueError('unverified A3 weights or cache')
    a3=torch.load(VERIFIED/'a3.pt',weights_only=True,mmap=True);source=json.loads((VERIFIED/'source_records.json').read_text());graphs={p.stem:ObjectGraph.restore(torch.load(p,weights_only=True)) for p in Path('runs/language_object_graph/graphs').glob('*.pt')}
    return a3,source,graphs,proof

def public(q,a3):
    return dict(static_logits=a3['logits'][q['query_index']].float().numpy(),grid_xy=a3['grid_xy'].float().numpy(),bounds=q['public_query']['world_bounds'],pose_world=q['pose_world'])

def primary_features(split):
    p=Path(f'runs/instance_grounding/features_{split}/features.pt');manifest=json.loads(p.with_name('manifest.json').read_text())
    if sha(p)!=manifest['feature_sha256']:raise ValueError('frozen RGB digest mismatch')
    f=torch.load(p,weights_only=True,mmap=True);primary={}
    for s in f['plan']['states']:
        m=s['metadata'];key=m['family_id']
        if key not in primary or m['cutoff_step']>primary[key]['metadata']['cutoff_step']:primary[key]=s
    return f,[primary[k] for k in sorted(primary)],manifest

def regions(state):
    m=state['metadata'];out={}
    for r in state['regions']:
        if r['source_step']>m['cutoff_step'] or r['episode_id']!=m['episode_id'] or r['map_name']!=m['map_name']:raise ValueError('future/cross-episode RGB')
        if r['native_min_side']<10 or r['depth_support']<.9:continue
        out[str(r['object_id'])]=r
    return out
