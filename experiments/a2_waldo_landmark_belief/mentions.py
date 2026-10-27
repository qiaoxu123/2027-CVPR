"""Map publicly linked landmark aliases to BERT full-instruction token spans.

Exact, case-insensitive word-boundary matching only. No GT identity, fuzzy
guessing or forced assignment of ambiguous aliases. Intended for fast-tokenizer
offsets; a missing or unresolved match remains masked/unknown.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np


def landmark_mention_spans(
    instruction: str,
    offsets: Sequence[Sequence[int]],
    references: Sequence[Mapping],
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean [L,T] BERT span masks and [L] verified-link flags.

    Each reference must carry:
      - linked: explicitly verified public instruction-to-map reference link
      - name: public linked landmark name
      - aliases: optional additional public strings
    The caller is responsible for matching the correct tokenizer+instruction.
    Missing offset maps, partial tokens and ambiguous shared mentions fail closed.
    """
    if not isinstance(instruction, str):
        raise ValueError("instruction must be text")
    pairs = np.asarray(offsets)
    if pairs.ndim != 2 or pairs.shape[-1] != 2:
        raise ValueError("fast-tokenizer character offsets required [T,2]")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("token offsets must be integers")
    if np.any(pairs < 0) or np.any(pairs[:, 1] < pairs[:, 0]):
        raise ValueError("invalid token offsets")
    t, l = len(pairs), len(references)
    masks = np.zeros((l, t), bool)
    eligible = np.zeros(l, bool)
    lower = instruction.casefold()
    candidate_spans = []
    for j, reference in enumerate(references):
        if not reference.get("linked", False):
            candidate_spans.append([])
            continue
        aliases = list(reference.get("aliases") or []) + [reference.get("name", "")]
        aliases = sorted({str(v).strip().casefold() for v in aliases if v and str(v).strip()},
                         key=lambda a: (-len(a), a))
        matches = []
        for alias in aliases:
            regex = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            matches.extend((m.start(), m.end()) for m in re.finditer(regex, lower))
        # Only maximal matched spans; shorter aliases nested in a longer name
        # cannot masquerade as a separate landmark mention.
        maximal = [span for span in sorted(set(matches), key=lambda p: (p[0], -(p[1]-p[0])))
                   if not any(other[0] <= span[0] and other[1] >= span[1] and other != span
                              for other in matches)]
        candidate_spans.append(maximal)

    for j, spans in enumerate(candidate_spans):
        if not spans:
            continue
        # If two references claim any overlapping character range, their
        # identities are unresolved: do not assign either mention.
        ambiguous = any(
            a < d and c < b
            for a, b in spans
            for k, others in enumerate(candidate_spans) if k != j
            for c, d in others
        )
        if ambiguous:
            continue
        for a, b in spans:
            overlap = (pairs[:, 1] > a) & (pairs[:, 0] < b) & (pairs[:, 1] > pairs[:, 0])
            if overlap.any():
                masks[j] |= overlap
        eligible[j] = masks[j].any()
    return masks, eligible
