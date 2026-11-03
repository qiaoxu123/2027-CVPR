"""Audit public relation availability and counterfactual text information loss."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import spacy
from graphnav.local_relation_metadata import relation_metadata
from scripts.parse_local_chain_roles import role, reference_phrases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier diagnostics')
    values = json.loads(args.roles.read_text())
    proposals = {key: relation_metadata(value) for key, value in values.items()}
    nlp = spacy.load('en_core_web_sm')
    controls = []
    for predicate in ['to the left of', 'to the right of', 'near', 'behind']:
        text = f'The white car {predicate} the grey car.'
        doc = nlp(text)
        parsed = role(doc, [])
        parsed['reference_phrases'] = reference_phrases(doc, parsed, [])
        controls.append({'instruction': text, 'target_text': parsed['target_text'],
                         'reference_phrases': parsed['reference_phrases'],
                         'relation': relation_metadata(parsed)})
    summary = {
        'tasks': len(values), 'active': sum(v['active'] for v in proposals.values()),
        'kind_counts': dict(Counter(v['kind'] for v in proposals.values() if v['active'])),
        'reason_counts': dict(Counter(v['reason'] for v in proposals.values())),
        'roles_sha256': hashlib.sha256(args.roles.read_bytes()).hexdigest(),
        'counterfactual_old_visual_text_identical': len({
            (v['target_text'], tuple(v['reference_phrases'])) for v in controls}) == 1,
        'counterfactual_new_kinds': [v['relation']['kind'] for v in controls],
        'private_labels_read': False,
        'warning': 'Syntax proposals only; neither reference visibility nor identity is verified.'}
    args.output.mkdir(parents=True)
    for name, data in [('summary', summary), ('proposals', proposals), ('counterfactuals', controls)]:
        (args.output / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
