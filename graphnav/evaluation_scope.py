"""Explicitly separate diagnostic subsets from official full evaluation."""

def validate_unified_scope(split,max_episodes,diagnostic_pilot):
    if split!='val_seen':raise ValueError('Unified-relation supports val_seen only')
    if diagnostic_pilot:
        if max_episodes is None or max_episodes<1:
            raise ValueError('Diagnostic pilot requires a positive explicit task budget')
    elif max_episodes is not None:
        raise ValueError('Unified-relation requires unfiltered FULL val_seen; opt into diagnostic pilot explicitly')
