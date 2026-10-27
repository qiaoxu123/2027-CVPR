"""Target-disjoint supervision splits and conservative public reference links.

Target identity is used ONLY by offline split construction. Public reference
resolution does not accept a target or inspect training labels.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from collections.abc import Mapping, Sequence


def _get(row, key):
    return row[key] if isinstance(row, Mapping) else getattr(row, key)


def split_target_groups(rows: Sequence, seed: int = 0) -> list[str]:
    """Stable 80/10/10 hash split of (map, target instance), independent of order.

    Fractions are expected fractions, not rounded per-map quotas. All queries
    and observation times for an instance have the same assignment.
    """
    assignments = []
    for row in rows:
        group = f"{seed}\0{_get(row,'map_name')}\0{_get(row,'object_id')}"
        fraction = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], 'big') / 2**64
        assignments.append('train' if fraction < .8 else 'dev' if fraction < .9 else 'calibration')
    return assignments


def _normalize(text: str) -> str:
    # Only formatting normalization; edit distance and semantic guessing are
    # deliberately excluded. Aliases must be supplied by the public source.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text)).strip()).casefold()


def public_reference_records(references: Sequence[Mapping]) -> list[dict]:
    """Merge same public instance and mask unresolved/ambiguous name identities.

    Input query_name is the actual source-provided instruction reference;
    name/aliases belong to the map object returned by the source resolver.
    A fuzzy resolver result alone cannot authorize a link.
    """
    merged = {}
    for ref in references:
        key = (ref['map_name'], ref['object_id'])
        canonical = [ref.get('name', '')] + list(ref.get('aliases') or [])
        accepted = { _normalize(v) for v in canonical if str(v).strip() }
        query = ref.get('query_name', ref.get('name', ''))
        verified = bool(_normalize(query)) and _normalize(query) in accepted
        if key not in merged:
            merged[key] = dict(ref, aliases=[], linked=False)
        record = merged[key]
        if verified:
            record['linked'] = True
            for alias in canonical + [query]:
                if alias and alias not in record['aliases']:
                    record['aliases'].append(alias)
    records = list(merged.values())
    claims = {}
    for i, record in enumerate(records):
        if record['linked']:
            for alias in record['aliases']:
                claims.setdefault(_normalize(alias), set()).add(i)
    ambiguous = {i for indices in claims.values() if len(indices) > 1 for i in indices}
    for i in ambiguous:
        records[i]['linked'] = False
    return records


def resolve_public_references(map_name: str, query_names: Sequence[str], named_objects: Sequence[Mapping]) -> list[dict]:
    """Resolve source-provided reference strings against public named map objects.

    Unlike the baseline catalog's largest-polygon shortcut, retain homonymous
    instances so ambiguity remains visible. Never inspect destination identity.
    """
    refs = []
    for query in query_names:
        matches = [obj for obj in named_objects if _normalize(obj.get('name', '')) == _normalize(query)]
        for obj in matches:
            refs.append(dict(map_name=map_name, object_id=int(obj['id']),
                             name=obj['name'], query_name=query,
                             contour=obj['contour'], semantic_description=obj.get('object_type', '')))
        if not matches:
            # Preserve an explicit unresolved record for funnel reporting.
            refs.append(dict(map_name=map_name, object_id=f'unresolved:{query}',
                             name='', query_name=query, contour=[], semantic_description=''))
    return public_reference_records(refs)


def validate_source_manifest(source: Mapping, dataset) -> None:
    """Bind stored indices/partitions to the live annotation before extraction."""
    annotation=hashlib.sha256(Path(dataset.annotation_path).read_bytes()).hexdigest()
    if annotation!=source['annotation_sha256']:
        raise ValueError('Live training annotation differs from audited annotation')
    if len(source['entries'])!=len(dataset):
        raise ValueError('Source population length changed')
    if {e['index'] for e in source['entries']}!=set(range(len(dataset))):
        raise ValueError('Source episode indices are missing or duplicated')
    for entry in source['entries']:
        row=dataset[entry['index']]
        actual=(row.map_name,row.object_id,row.description_id)
        expected=(entry['map_name'],entry['object_id'],entry['description_id'])
        if actual!=expected:
            raise ValueError(f'Episode identity mismatch at index {entry["index"]}')
        if split_target_groups([row],seed=0)[0]!=entry['split']:
            raise ValueError('Target group assignment differs from fixed seed0 protocol')
