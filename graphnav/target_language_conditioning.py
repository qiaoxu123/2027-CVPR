"""Explicit training-data overlay; original default and reference roles preserved."""


def condition_target_language(rows, mode='role'):
    if mode not in ('role', 'full-instruction'):
        raise ValueError('Unknown target language mode')
    result = []
    for row in rows:
        copied = dict(row)
        if mode == 'full-instruction':
            if 'full_text' not in row:
                raise ValueError('Fresh full-instruction vector required')
            if row['full_text'].shape != row['target_text'].shape:
                raise ValueError('Target/full language dimension mismatch')
            copied['target_text'] = row['full_text']
        result.append(copied)
    return result
