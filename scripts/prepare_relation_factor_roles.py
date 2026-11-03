"""Parse current public TRAIN text into distinct reference/anchor proposals."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import spacy
from scripts.analyze_instruction_landmark_association import PublicNames
from scripts.parse_local_chain_roles import role
from graphnav.relation_factor_roles import build_relation_roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('public-frames', 'catalog', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing role graphs')
    frames = [json.loads(line) for line in args.public_frames.read_text().splitlines()]
    queries = {f['task_key']: (f['map_name'], f['instruction']) for f in frames}
    names = PublicNames(json.loads(args.catalog.read_text()))
    nlp = spacy.load('en_core_web_sm')
    results = {}
    for key, (map_name, text) in queries.items():
        doc = nlp(text); mentions = names.match(map_name, text)
        parsed = role(doc, mentions)
        results[key] = build_relation_roles(doc, parsed, mentions)
        results[key]['map_name'] = map_name
    summary = dict(tasks=len(results), active_tasks=sum(bool(r['references']) for r in results.values()),
                   multiple_reference_tasks=sum(len(r['references']) > 1 for r in results.values()),
                   map_anchored_tasks=sum(any(ref['anchors'] for ref in r['references']) for r in results.values()),
                   intermediate_anchored_tasks=sum(any(ref['anchors'] and not ref['public_ids'] for ref in r['references']) for r in results.values()),
                   ordinary_reference_tasks=sum(any(not ref['public_ids'] for ref in r['references']) for r in results.values()),
                   reference_counts=dict(Counter(ref['kind'] for r in results.values() for ref in r['references'])),
                   private_labels_read=False, spacy_version=spacy.__version__, spacy_model_version=nlp.meta['version'])
    for label, path in [('public_frames', args.public_frames), ('catalog', args.catalog),
                        ('parser', Path(__file__)), ('codec', Path('graphnav/relation_factor_roles.py'))]:
        summary[label+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    (args.output/'roles.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
