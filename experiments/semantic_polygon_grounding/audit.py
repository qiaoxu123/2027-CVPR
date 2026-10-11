"""Read-only P4 schema/name audit and identifiable public-name Oracle replay.

No target-nearest reference selection. Target coordinates are used only in
offline Goal metrics. Anonymous/private target polygons never enter candidates.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from shapely.geometry import Polygon

from graphnav.landmark_alias import build_name_index, canonical_name, resolve_references
from graphnav.semantic_geometry import build_entities, parse_spatial_relations
from experiments.bidirectional_core.stats import summarize_rows

DATA = Path('/home/rental/20260922_1/Workspace/DATA/sbfnav-data')
OUT = Path('runs/semantic_polygon_grounding')
PROTOCOL = Path(__file__).with_name('protocol.json')
COLORS = r'white|gray|grey|red|blue|brown|black|green|yellow|orange|beige'
PATTERNS = {
    'color': rf'\b(?:{COLORS})\b',
    'type': r'\b(?:building|house|church|road|street|car|vehicle|parking|footpath|bridge|hall|court|school|shop|store)\b',
    'shape': r'\b(?:rectangular|square|round|circular|l-shaped|u-shaped|triangular|long|narrow|wide)\b',
    'size': r'\b(?:small|large|big|tiny|huge|tall|short)\b',
    'distance': r'\b(?:near|next to|beside|close to|far|distant|metres?|meters?|\d+\s*m)\b',
    'direction': r'\b(?:north(?:east|west)?|south(?:east|west)?|east|west|left|right|front|behind|back|opposite|across)\b',
    'between': r'\bbetween\b',
    'material': r'\b(?:brick|concrete|metal|glass|wooden|stone)\b',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def valid_polygon(obj):
    p = np.asarray(obj.get('contour', []), dtype=float)
    if p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 3 or not np.isfinite(p).all():
        return None
    q = Polygon(p)
    return q if q.is_valid and q.area > 1e-10 else None


def geometry_summary(obj):
    q = valid_polygon(obj)
    if q is None:
        return dict(valid_polygon=False, area_m2=None, orientation_deg=None,
                    aspect_ratio=None, centroid_xy=None)
    corners = np.asarray(q.minimum_rotated_rectangle.exterior.coords)[:4]
    delta = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(delta, axis=1)
    i = int(lengths.argmax())
    return dict(valid_polygon=True, area_m2=float(q.area),
                orientation_deg=float(np.rad2deg(np.arctan2(delta[i, 1], delta[i, 0])) % 180),
                aspect_ratio=float(lengths.max() / max(lengths.min(), 1e-10)),
                centroid_xy=[float(q.centroid.x), float(q.centroid.y)])


def name_binding(name, objects, index):
    key = canonical_name(name)
    exact = index['exact'].get(key, [])
    typed = index['typed'].get(key, [])
    matches = resolve_references([name], objects, index=index, max_landmarks=10000,
                                 match_mode='fuzzy')
    mode = 'exact' if exact else 'typed' if typed else 'fuzzy' if matches else 'unresolved'
    valid_ids = [x['id'] for x in matches if valid_polygon(objects[x['id']]) is not None]
    return dict(phrase=name, mode=mode, candidate_ids=[x['id'] for x in matches],
                valid_ids=valid_ids, unique_valid=len(valid_ids) == 1 and len(matches) == 1,
                reference_instance_truth=None,
                truth_status='name_only_annotation_no_reference_instance_id')


def anonymous_reference_flags(instruction):
    """Conservative trigger audit; no identity/attribute labels are fabricated."""
    noun = r'(?:building|house|church|car|vehicle|parking lot|car park|road|street|footpath|bridge)'
    modifiers = rf'(?:(?:{COLORS}|small|large|big|long|rectangular|square|round|brick|roofed)\s+){{0,5}}'
    pattern = rf'\b(?:near|beside|next to|behind|in front of|between|left of|right of|east of|west of|north of|south of)\s+(?:the\s+|a\s+|an\s+)?({modifiers}{noun})\b'
    return [m.group(1) for m in re.finditer(pattern, instruction.lower())]


def annotation_reference_names(processed, map_name, target_id, annotation_id):
    """Offline loader only: fields selected via private annotation keys."""
    annotations = processed.get(map_name, {}).get(str(target_id), [])
    if annotation_id >= len(annotations):
        return []
    return list(dict.fromkeys(map(str, annotations[annotation_id].get('landmarks', []))))


def inspect_query(instruction, names, map_name, objects, indices):
    bindings = [name_binding(n, objects[map_name], indices[map_name]) for n in names]
    # Removing known map-name spans avoids counting an explicit named anchor as anonymous.
    text = canonical_name(instruction)
    for b in sorted(bindings, key=lambda b: len(b['phrase']), reverse=True):
        text = re.sub(r'\b' + re.escape(canonical_name(b['phrase'])) + r'\b', ' namedanchor ', text)
        for oid in b['candidate_ids']:
            actual = canonical_name(objects[map_name][oid].get('name', ''))
            if actual:
                text = re.sub(r'\b' + re.escape(actual) + r'\b', ' namedanchor ', text)
    anonymous = anonymous_reference_flags(text)
    unique = bool(bindings) and all(b['unique_valid'] for b in bindings)
    multi = any(len(b['candidate_ids']) > 1 for b in bindings)
    unresolved = any(not b['candidate_ids'] for b in bindings)
    flags = {k: bool(re.search(p, instruction.lower())) for k, p in PATTERNS.items()}
    flags['multiple_named_anchors'] = len(bindings) >= 2
    flags['anonymous_reference_trigger'] = bool(anonymous)
    return dict(bindings=bindings, all_named_bindings_unique_valid=unique,
                multi_named=multi, unresolved_named=unresolved,
                anonymous_reference_phrases=anonymous, lexical_flags=flags,
                declared_names_not_literal=[n for n in names if canonical_name(n) not in canonical_name(instruction)],
                names=[] if not names else names,
                identifiable_ambiguous_anchor_truth=False)


def map_audit(objects, public):
    rows = []; counts = Counter(); types = defaultdict(Counter)
    for m, entries in objects.items():
        for oid, o in entries.items():
            g = geometry_summary(o); named = bool(canonical_name(o.get('name', '')))
            id_valid = str(o.get('id')) == str(oid) and o.get('map_name') == m
            counts['objects'] += 1; counts['named'] += named; counts['unnamed'] += not named
            counts['valid_polygon'] += g['valid_polygon']; counts['id_map_consistent'] += id_valid
            counts['named_valid_polygon'] += named and g['valid_polygon']
            types[o.get('object_type', 'Unknown')]['objects'] += 1
            types[o.get('object_type', 'Unknown')]['named'] += named
            types[o.get('object_type', 'Unknown')]['valid_polygon'] += g['valid_polygon']
            rows.append(dict(map_name=m, object_id=str(oid), name=o.get('name', ''),
                             object_type=o.get('object_type'), **g, id_map_consistent=id_valid,
                             inference_geometry_allowed=str(oid) in public.get(m, {}),
                             descriptions_allowed_for_inference=False))
    duplicates = []
    for m, entries in public.items():
        groups = defaultdict(list)
        for oid, o in entries.items(): groups[canonical_name(o.get('name', ''))].append(str(oid))
        for name, ids in groups.items():
            if len(ids) < 2: continue
            edges = []; overlap = 0; types_here = sorted({entries[i]['object_type'] for i in ids})
            for j, a in enumerate(ids):
                ga = valid_polygon(entries[a])
                for b in ids[j+1:]:
                    gb = valid_polygon(entries[b])
                    if ga is not None and gb is not None and ga.intersects(gb):
                        edges.append((a, b)); overlap += ga.intersection(gb).area > 0
            parent = {i: i for i in ids}
            def find(a):
                while parent[a] != a: a = parent[a]
                return a
            for a, b in edges: parent[find(a)] = find(b)
            duplicates.append(dict(map_name=m, canonical_name=name, ids=ids, types=types_here,
                                   touching_components=len({find(i) for i in ids}),
                                   positive_area_overlap_pairs=overlap,
                                   semantic_instance_truth=None,
                                   note='disconnected is diagnostic, not proof of distinct semantic referents'))
    return dict(counts=dict(counts), types={k: dict(v) for k,v in types.items()},
                maps=len(objects), public_named_polygons=sum(map(len, public.values())),
                homonymous_name_groups=len(duplicates)), rows, duplicates


def raw_split_audit(split, objects, indices, processed):
    path = DATA / f'processed_citynav/citynav_{split}.json'
    episodes = json.loads(path.read_text()); counts = Counter(); rows = []; seen = set()
    for e in episodes:
        m = f"{e['area']}_block_{e['block']}"; oid = str(e['object_ids'][0]); ann = int(e['ann_ids'][0])
        key = (m, oid, ann)
        if key in seen: continue
        seen.add(key)
        instruction = e['descriptions'][0]
        names = annotation_reference_names(processed, m, oid, ann)
        q = inspect_query(instruction, names, m, objects, indices)
        counts['independent_map_target_instruction'] += 1
        for k in ['all_named_bindings_unique_valid', 'multi_named', 'unresolved_named']:
            counts[k] += q[k]
        counts['no_named_anchor'] += not bool(names)
        counts['anonymous_reference_trigger'] += bool(q['anonymous_reference_phrases'])
        counts['declared_name_not_literal_queries'] += bool(q['declared_names_not_literal'])
        counts['target_is_also_named_reference'] += any(oid in b['candidate_ids'] for b in q['bindings'])
        for k,v in q['lexical_flags'].items(): counts['lexical_' + k] += v
        for b in q['bindings']: counts['mention_' + b['mode']] += 1
        rows.append(dict(map_name=m, annotation_key=[oid, ann], instruction=instruction, **q))
    return dict(episodes=len(episodes), unique_queries=len(seen), counts=dict(counts), source_sha256=sha(path)), rows


def goal_metrics(logits, normalized_grid, bounds, target):
    grid = np.asarray(bounds[:2]) + np.asarray(normalized_grid) * (np.asarray(bounds[2:])-bounds[:2])
    # Exactly preserve P2/P2.5 Goal ranking, including high-index tie order.
    order = np.argsort(np.asarray(logits), kind='stable')[::-1]
    distances = np.linalg.norm(grid - np.asarray(target), axis=1)
    positive = distances <= 20.
    hit = np.flatnonzero(positive[order])
    return dict(goal_r1=float(positive[order[0]]), goal_r4=float(positive[order[:4]].any()),
                goal_r8=float(positive[order[:8]].any()), ne=float(distances[order[0]])), order.tolist(), int(hit[0]+1) if len(hit) else None


def credible(region):
    return (float(region.get('depth_support', 0)) >= .9 and
            float(region.get('native_min_side', 0)) >= 16)


def replay(split, objects, indices, processed, public):
    path = Path(f'runs/instance_grounding/plan_{split}/plan.pt')
    manifest = json.loads(path.with_name('manifest.json').read_text())
    assert sha(path) == manifest['files']['plan.pt']
    plan = torch.load(path, map_location='cpu', weights_only=True)
    rows=[]; primary={}; query_cache={}
    for s in plan['states']:
        m=s['metadata']; sid=m['sample_id']; family=m['family_id']; label=plan['labels'][sid]
        # Offline annotation lookup uses original referring keys; the target ID
        # does not enter the public name resolver or geometry features.
        f=family.split(':'); ann=int(f[3]); map_name=m['map_name']
        key=(map_name,str(label['target_object_id']),ann,m['instruction'])
        if key not in query_cache:
            names=annotation_reference_names(processed,map_name,label['target_object_id'],ann)
            query_cache[key]=inspect_query(m['instruction'],names,map_name,objects,indices)
        q=query_cache[key]; p=s['public']
        metrics, ranking, target_rank=goal_metrics(p['static_logits'],p['grid_xy'],p['bounds'],label['target_world'])
        regions={str(r['object_id']):r for r in s['regions']}
        for rg in regions.values():
            assert rg['source_step']<=m['cutoff_step']
        candidates=[]
        for b in q['bindings']:
            cs=[]
            for oid in b['candidate_ids']:
                obj=objects[map_name][oid];g=geometry_summary(obj)
                rg=regions.get(oid)
                cs.append(dict(object_id=oid,name=obj['name'],object_type=obj['object_type'],**g,
                               probability_name_set=1/len(b['candidate_ids']),
                               semantic_instance_probability=None,
                               observed=rg is not None,credible_observed=rg is not None and credible(rg),
                               observed_step=None if rg is None else rg['source_step'],
                               rgb_attribute_status='Unknown; no reference attribute label',
                               source='public_named_landmark_polygon'))
            candidates.append(dict(phrase=b['phrase'],mode=b['mode'],candidates=cs,
                                   reference_instance_truth=None))
        observed_ambiguous=any(len(b['candidate_ids'])>1 and
                               sum(oid in regions and credible(regions[oid]) for oid in b['candidate_ids'])>=2
                               for b in q['bindings'])
        full=resolve_references(q['names'],objects[map_name],index=indices[map_name],max_landmarks=10000,match_mode='fuzzy')
        capped=resolve_references(q['names'],objects[map_name],index=indices[map_name],max_landmarks=16,match_mode='fuzzy')
        # Normalize public reference contours exactly as runtime; no target contour.
        bounds=np.asarray(p['bounds']); span=bounds[2:]-bounds[:2]
        normalized=[dict(x,contour=((np.asarray(x['contour'])-bounds[:2])/span).tolist()) for x in capped]
        entities=build_entities(normalized,[],bounds.tolist())
        relations=parse_spatial_relations(m['instruction'],entities)
        named_sensitive=q['multi_named'] or q['unresolved_named'] or len(full)>len(capped)
        any_sensitive=named_sensitive or bool(q['anonymous_reference_phrases']) or not q['names']
        # A known-name-set oracle has exactly the same reference inputs as A0.
        # It provides no supervision for choosing among homonymous instances.
        oracle=dict(metrics,recovery=0.,damage=0.,net_recovery=0.)
        row=dict(sample_id=sid,family_id=family,map_name=map_name,split=split,
                 cutoff_step=m['cutoff_step'],instruction=m['instruction'],bindings=candidates,
                 audit=q,metrics=metrics,geometry_ranking=ranking,target_rank=target_rank,
                 known_name_oracle_metrics=oracle,known_name_oracle_ranking=ranking,
                 strict_unique_reference_oracle_eligible=q['all_named_bindings_unique_valid'],
                 ambiguous_instance_oracle_metrics=None,
                 oracle_status='identical_public_name_set; ambiguous_instance_oracle_unavailable',
                 named_binding_sensitive=named_sensitive,any_reference_sensitive=any_sensitive,
                 multiple_credible_same_named_candidates=observed_ambiguous,
                 independently_evaluable_ambiguous_reference=False,
                 observed_public_named_rois=sum(oid in public.get(map_name,{}) for oid in regions),
                 observed_nonpublic_annotation_rois=sum(oid not in public.get(map_name,{}) for oid in regions),
                 capped_references=len(full)>len(capped),full_reference_count=len(full),
                 parsed_relations=[dict(kind=r.kind,anchors=[list(g) for g in r.anchors]) for r in relations],
                 target_also_reference_offline_flag=any(str(label['target_object_id']) in b['candidate_ids'] for b in q['bindings']))
        rows.append(row)
        if family not in primary or m['cutoff_step']>primary[family]['cutoff_step']:
            primary[family]=row
    primary=list(primary.values())
    return rows,primary,dict(plan_sha256=sha(path),source_manifest_sha256=sha(path.with_name('manifest.json')),
                             plan_states=len(rows),families=len(primary),maps=manifest['maps'],
                             independently_evaluable_ambiguous_anchor_families=0)


def aggregate(rows):
    base={k:summarize_rows(rows,k,seed=20261011) for k in ['goal_r1','goal_r4','goal_r8','ne']}
    subsets={}
    for key in ['strict_unique_reference_oracle_eligible','named_binding_sensitive',
                'any_reference_sensitive','multiple_credible_same_named_candidates',
                'target_also_reference_offline_flag','capped_references']:
        chosen=[r for r in rows if r[key]]
        subsets[key]=dict(states=len(chosen),families=len({r['family_id'] for r in chosen}),
                          static_errors=sum(not r['metrics']['goal_r1'] for r in chosen),
                          static_error_families=len({r['family_id'] for r in chosen if not r['metrics']['goal_r1']}),
                          goal_r1=summarize_rows(chosen,'goal_r1',seed=20261011))
    deltas=[dict(sample_id=r['sample_id'],family_id=r['family_id'],map_name=r['map_name'],
                 metrics=dict(goal_r1=0.,recovery=0.,damage=0.,net_recovery=0.)) for r in rows]
    ceilings={}
    for flag in ['named_binding_sensitive','any_reference_sensitive']:
        changed=[dict(sample_id=r['sample_id'],family_id=r['family_id'],map_name=r['map_name'],
                      metrics={'hypothetical_goal_delta':float(bool(r[flag]) and not r['metrics']['goal_r1'])}) for r in rows]
        ceilings[flag]=summarize_rows(changed,'hypothetical_goal_delta',seed=20261011)
    return dict(A0_geometry=base,
                A1_known_name_set=dict(base,paired_delta={k:summarize_rows(deltas,k,seed=20261011) for k in deltas[0]['metrics']}),
                strict_unique_oracle=dict(n=subsets['strict_unique_reference_oracle_eligible']['states'],
                                         delta_r1=0.,note='unique annotated name-to-ID already bound; no input changes'),
                ambiguous_instance_oracle=None,subsets=subsets,eligibility_only_upper_bound=ceilings,
                independently_evaluable_ambiguous_anchor_families=0,
                per_map={m:dict(n=len([r for r in rows if r['map_name']==m]),
                               errors=sum(not r['metrics']['goal_r1'] for r in rows if r['map_name']==m))
                         for m in sorted({r['map_name'] for r in rows})})


def run():
    if OUT.exists(): raise ValueError('Completed/partial P4 output is immutable; choose a new run directory')
    OUT.mkdir(parents=True);start=time.time();torch.set_num_threads(1)
    objects=json.loads((DATA/'cityrefer/objects.json').read_text())
    public_raw=json.loads((DATA/'cityrefer/landmarks.json').read_text())
    # Explicit whitelist strips private descriptions and target-centric boxes.
    public={m:{str(oid):{k:o[k] for k in ['id','map_name','name','object_type','contour']}
               for oid,o in a.items()} for m,a in public_raw.items()}
    assert all(o['name'] for a in public.values() for o in a.values())
    assert all(str(oid) in objects[m] and objects[m][oid]['contour']==o['contour'] for m,a in public.items() for oid,o in a.items())
    indices={m:build_name_index(a) for m,a in public.items()}
    processed=json.loads((DATA/'cityrefer/processed_descriptions.json').read_text())
    ma,mrows,duplicates=map_audit(objects,public);save(OUT/'map_audit.json',ma);save(OUT/'object_geometry_audit.json',mrows);save(OUT/'homonyms.json',duplicates)
    annotation_fields=Counter(tuple(sorted(d)) for a in processed.values() for ds in a.values() for d in ds)
    save(OUT/'annotation_schema.json',dict(schema={','.join(k):v for k,v in annotation_fields.items()},
                                         reference_instance_ids_present=any(any('id' in k for k in keys) for keys in annotation_fields),
                                         named_public_whitelist_only=True,private_objects_not_assumed_public=True))
    raw={};metrics={};sources={}
    for split in ['train_seen','val_seen']:
        raw[split],queries=raw_split_audit(split,public,indices,processed)
        with gzip.open(OUT/(split+'_query_audit.json.gz'),'wt') as f:json.dump(queries,f,ensure_ascii=False,allow_nan=False)
    # Configuration is already frozen on disk, prior to the fixed Seen replay.
    for split in ['train','dev','seen']:
        rows,primary,sources[split]=replay(split,public,indices,processed,public)
        metrics[split]=dict(primary_family=aggregate(primary),all_prefixes=aggregate(rows))
        for name,rs in [('all_prefix_predictions',rows),('family_predictions',primary)]:
            with gzip.open(OUT/f'{split}_{name}.json.gz','wt') as f:json.dump(rs,f,ensure_ascii=False,allow_nan=False)
        print(json.dumps(dict(split=split,states=len(rows),families=len(primary),elapsed=time.time()-start)),flush=True)
    train_maps=set(sources['train']['maps']);dev_maps=set(sources['dev']['maps']);assert not(train_maps&dev_maps)
    save(OUT/'raw_split_audit.json',raw);save(OUT/'metrics.json',metrics)
    save(OUT/'decision.json',dict(continue_research=False,first_gate_pass=False,independent_ambiguous_anchor_truth_families=0,
                                  instance_anchor_oracle_evaluable=False,known_name_set_oracle_delta=0,
                                  A2_A3_A4_A5='not run: absent independently verifiable reference-instance supervision',
                                  training=False,unseen=False,geometry_changed=False,private_polygon_inference=False))
    save(OUT/'execution.json',dict(seconds=time.time()-start,seed=20261011,sources=sources,
                                   protocol_sha256=sha(PROTOCOL),train_dev_shared_maps=sorted(train_maps&dev_maps),
                                   resource_sha256={str(p):sha(p) for p in [DATA/'cityrefer/objects.json',DATA/'cityrefer/landmarks.json',DATA/'cityrefer/processed_descriptions.json']},
                                   gpu_encoding=0,training=0,reference_instance_labels=0))
    print(json.dumps(dict(complete=True,seconds=time.time()-start)),flush=True)


if __name__=='__main__':run()
