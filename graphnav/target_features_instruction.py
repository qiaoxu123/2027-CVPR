"""Conservative target/attributes/anchor/relation text for the existing network.

No guessed entity roles or reordered clauses: preserve all target identity,
negation, ordinal, intermediate-reference and Stop text in original order.
This is not a learned semantic parser. Only an initial navigation imperative
can be removed; ambiguous inputs pass through unchanged. No annotation API.
"""
import re

VERSION = 'content-preserving-v1'
_PREFIX = re.compile(r'^\s*(?:navigate|go|fly|head|travel)\s+to\s+(?=\S)', re.I)


def content_instruction(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Expected nonempty original public instruction')
    return _PREFIX.sub('', text, count=1)
