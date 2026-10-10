"""Keep only this instruction's supplied resolved landmark bindings, without GT."""
from .landmark_alias import canonical_name


def instruction_landmarks_only(record, reference_names):
    allowed = {canonical_name(name) for name in reference_names if canonical_name(name)}
    kept, removed = [], []
    for landmark in record['landmarks']:
        identities = {canonical_name(name) for name in
                      [landmark.get('name', ''), *landmark.get('aliases', [])]}
        (kept if identities & allowed else removed).append(landmark)
    audit = dict(control='instruction-bound-landmarks-only', target_used=False,
        original_polygons=len(record['landmarks']), retained_polygons=len(kept),
        reference_names=list(reference_names),
        retained_names=list(dict.fromkeys(row['name'] for row in kept)),
        removed_names=list(dict.fromkeys(row['name'] for row in removed)),
        note='Names/aliases traceable to this instruction supplied CityRefer binding. Preserve legitimate shorthand, fuzzy aliases and same-name polygons. No count limit or target-distance filtering.')
    return dict(record, landmarks=kept), audit
