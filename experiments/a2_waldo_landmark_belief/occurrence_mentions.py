"""Opt-in exact public-name mentions resolved per occurrence, not per sentence.

No inference of entity identity: equal/crossing claims stay ambiguous. A longer
literal public name suppresses a strictly nested substring at that occurrence
only. Legacy V3's matcher remains unchanged.
"""
import re
from collections.abc import Mapping, Sequence

import numpy as np

from .mentions import landmark_mention_spans


VERSION = 'public-mention-occurrence-v1'


def occurrence_mention_spans(instruction: str, offsets: Sequence[Sequence[int]],
                             references: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
    """Return [reference,token] masks and flags with literal longest-name scope.

    Resolve text containment before token coverage: a truncated long public
    name must never be relabeled as its fully visible shorter substring.
    """
    # Share all legacy text/offset input validation without changing its code.
    landmark_mention_spans(instruction, offsets, [])
    pairs = np.asarray(offsets)
    masks = np.zeros((len(references), len(pairs)), dtype=bool)
    candidates = set()
    for index, reference in enumerate(references):
        if not reference.get('linked', False):
            continue
        aliases = [reference.get('name', ''), *(reference.get('aliases') or [])]
        for alias in {str(a).strip() for a in aliases if a and str(a).strip()}:
            for match in re.finditer(r'(?<!\w)'+re.escape(alias)+r'(?!\w)', instruction, re.IGNORECASE):
                candidates.add((index, match.start(), match.end()))
    maximal = [(index, a, b) for index,a,b in sorted(candidates)
               if not any(c <= a and d >= b and (c,d) != (a,b)
                          for _,c,d in candidates)]
    for index,a,b in maximal:
        # Reject this occurrence only; another independent literal appearance
        # of the same verified reference can still supply evidence.
        if any(other != index and a < d and c < b for other,c,d in maximal):
            continue
        overlap = (pairs[:,1] > a) & (pairs[:,0] < b) & (pairs[:,1] > pairs[:,0])
        covered = np.zeros(b-a, dtype=bool)
        for start,end in pairs[overlap]:
            covered[max(start,a)-a:min(end,b)-a] = True
        required = np.array([not ch.isspace() for ch in instruction[a:b]], dtype=bool)
        if overlap.any() and covered[required].all():
            masks[index] |= overlap
    return masks, masks.any(axis=1)
