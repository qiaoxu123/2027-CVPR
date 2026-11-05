"""Conservative direct-relation eligibility; public text/catalog only.

Dependency proposals remain proposals. Full original BERT tokens, rather than
these discrete predicates alone, condition the trainable network.
"""
import re
from shapely.geometry import Polygon
from .unified_instruction import _public_landmark

PREDICATES = ('north', 'south', 'east', 'west', 'near', 'inside', 'outside', 'along')
ALIASES = {
    'north of': 'north', 'south of': 'south', 'east of': 'east', 'west of': 'west',
    'near': 'near', 'next to': 'near', 'adjacent to': 'near', 'close to': 'near',
    'beside': 'near', 'inside': 'inside', 'in': 'inside', 'within': 'inside',
    'outside': 'outside', 'along': 'along',
}
FATAL = {'relation_scope_unresolved', 'target_role_unresolved',
         'target_category_ambiguous_or_unknown', 'target_lexical_fallback_scope_uncertain',
         'earlier_target_category_noun_scope_requires_review', 'public_alias_multiple_names'}
INTERMEDIATE = re.compile(
    r'\b(?:parking\s+(?:lot|area|space|strip)|row|rows|column|line|road|street|'
    r'houses|buildings|vehicles|cars|lot|intersection|junction)\b', re.I)


def select_direct_relations(parsed, public_objects):
    """No private label accepted. Return a rejection ledger, not silent fallback."""
    if parsed.get('target_GT_used') is not False:
        raise ValueError('Parser must explicitly declare GT-free provenance')
    text = parsed['original_text']
    fatal = sorted(set(parsed.get('ambiguity', [])) & FATAL)
    result = dict(original_instruction=text, target=parsed.get('target'),
                  relations=[], rejected=[], target_GT_used=False,
                  status='automatic_scope_checked_not_human_gold')
    if fatal:
        result['rejected'].append(dict(reason='global_scope_ambiguity', flags=fatal))
        return result
    target = parsed.get('target') or {}
    entities = {e['id']: e for e in parsed.get('entities', [])}
    target_entity = entities.get(target.get('entity'))
    if not target_entity:
        result['rejected'].append(dict(reason='target_entity_missing'))
        return result
    edges = {e['id']: e for e in parsed.get('relations', [])}
    seen = set()
    for path in parsed.get('direct_relations', []):
        reason = None
        if len(path.get('relations', [])) != 1 or len(path.get('landmark_names', [])) != 1:
            result['rejected'].append(dict(reason='non_unique_direct_path'))
            continue
        ref = path['relations'][0]
        edge = edges.get(ref['relation_id'], {})
        phrase = edge.get('predicate', '').lower()
        if (ref.get('traversal') != 'forward' or edge.get('subject') != target['entity']):
            reason = 'relation_role_or_traversal_uncertain'
        elif phrase not in ALIASES:
            reason = 'unsupported_or_unresolved_reference_frame'
        elif edge.get('relation_kind') != 'spatial_proposal' or edge.get('ambiguity'):
            reason = 'not_reliable_spatial_edge'
        mentions = [m for m in parsed.get('landmarks', [])
                    if m.get('public_names') == path['landmark_names']]
        if len(mentions) != 1:
            reason = reason or 'mention_binding_not_unique'
        if reason:
            result['rejected'].append(dict(reason=reason, phrase=phrase))
            continue
        mention = mentions[0]
        # Require the selected predicate to govern this exact public-name span.
        before = text[:mention['start']]
        match = re.search(r'\b' + re.escape(phrase) + r'\s+(?:the\s+)?$', before, re.I)
        if not match:
            reason = 'predicate_not_immediately_bound_to_public_name'
        elif re.match(r"(?:['’]s)\s+(?:parking|lot|road|building|house)\b",
                      text[mention['end']:], re.I):
            reason = 'possessive_intermediate_region'
        else:
            start, end = target_entity['span']
            if start < match.start():
                scope = text[end:match.start()]
                # Repeated explicit target mentions are retained, not treated as
                # a new intermediate entity. Other category nouns are not erased.
                scope = re.sub(r'\b(?:this|that|the)\s+' +
                               re.escape(target_entity['text']) + r'\b', '', scope, flags=re.I)
                if INTERMEDIATE.search(scope):
                    reason = 'intermediate_entity_in_relation_scope'
                for entity in entities.values():
                    a, b = entity['span']
                    if entity['id'] == target['entity'] or not end <= a < match.start():
                        continue
                    if entity['text'].lower() in {'roof', 'wall', 'walls', 'window', 'door'}:
                        continue
                    repeat = (entity['text'].lower() == target_entity['text'].lower()
                              and re.search(r'\b(?:this|that)\s+$', text[:a], re.I))
                    if not repeat:
                        reason = 'intermediate_entity_in_relation_scope'
            else:
                # Fronted clauses need independently resolved coreference. Do
                # not silently use a nearest noun heuristic for training labels.
                reason = 'fronted_relation_scope_requires_review'
        landmark, error = _public_landmark(path['landmark_names'][0], public_objects)
        if error:
            reason = reason or error
        if landmark and len(landmark['public_ids']) > 16:
            reason = reason or 'anchor_exceeds_geometry_capacity'
        if landmark and any(not Polygon(public_objects[str(oid)]['contour']).is_valid
                            or Polygon(public_objects[str(oid)]['contour']).area <= 0
                            for oid in landmark['public_ids']):
            reason = reason or 'invalid_public_polygon'
        if reason:
            result['rejected'].append(dict(reason=reason, phrase=phrase,
                                          landmark=path['landmark_names'][0]))
            continue
        key = (landmark['name'], ALIASES[phrase])
        if key in seen:
            continue
        seen.add(key)
        result['relations'].append(dict(
            predicate=ALIASES[phrase], predicate_id=PREDICATES.index(ALIASES[phrase]),
            original_predicate=phrase, landmark=landmark,
            source_span=edge['source_span'], source_text=edge['source_text'],
            target_entity=target['entity'], reference_entity=edge['object'],
            reference_frame=edge['reference_frame']['frame'],
            coordinate_convention=('engineering_world_XY_cardinal_unverified_geographic'
                                   if ALIASES[phrase] in PREDICATES[:4] else 'metric_topology'),
        ))
    return result
