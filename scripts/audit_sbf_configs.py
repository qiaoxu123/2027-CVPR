"""Read-only SBF model/suite configuration audit without changing old branches."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


def configuration_kind(raw):
    if not isinstance(raw, dict):
        raise ValueError('configuration must be a mapping')
    if '_base_' in raw or ('model' in raw and 'data' in raw):
        return 'model'
    required = {'seed', 'splits', 'episodes_per_split', 'variants'}
    if required <= raw.keys():
        if set(raw) != required or not isinstance(raw['seed'], int):
            raise ValueError('invalid evaluation suite schema')
        if not raw['splits'] or any(s not in ('val_seen', 'val_unseen') for s in raw['splits']):
            raise ValueError('suite must use approved validation splits')
        if not isinstance(raw['episodes_per_split'], int) or raw['episodes_per_split'] < 1:
            raise ValueError('invalid suite denominator')
        names = set()
        for variant in raw['variants']:
            if set(variant) != {'name', 'overrides'} or not isinstance(variant['name'], str) or not variant['name'] or variant['name'] in names:
                raise ValueError('invalid or duplicate suite variant')
            names.add(variant['name'])
            if not isinstance(variant['overrides'], list) or any(not isinstance(v, str) for v in variant['overrides']):
                raise ValueError('invalid typed overrides')
        if not names:
            raise ValueError('empty evaluation suite')
        return 'suite'
    raise ValueError('unknown configuration schema')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); source = Path(args.source).resolve(); output = Path(args.output)
    if output.exists():
        raise ValueError('refusing audit overwrite')
    sys.path.insert(0, str(source))
    from sbfnav.config import load_config, apply_overrides, validate_config, resolved_config_sha256
    root = source/'configs/sbfnav'; baseline = load_config(root/'main.yaml'); validate_config(baseline)
    rows = []
    for path in sorted(root.rglob('*.yaml')):
        raw = yaml.safe_load(path.read_text()); kind = configuration_kind(raw)
        configs = [('model', load_config(path))] if kind == 'model' else [
            (v['name'], apply_overrides(baseline, v['overrides'])) for v in raw['variants']]
        for _, config in configs:
            validate_config(config)
            if config['data']['allow_test_unseen']:
                raise ValueError('unsealed test split')
        rows.append(dict(path=str(path.relative_to(source)), kind=kind,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            resolved_sha256={name:resolved_config_sha256(config) for name, config in configs}))
    output.parent.mkdir(parents=True, exist_ok=True)
    result=dict(source=str(source), files=len(rows), model_configs=sum(r['kind']=='model' for r in rows),
        suite_configs=sum(r['kind']=='suite' for r in rows), rows=rows,
        note='Read-only schema-aware audit. Historical external glob test remains failed; this audit validates model configs and every suite variant separately rather than passing suite YAML to the model validator.')
    output.write_text(json.dumps(result, indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__ == '__main__':
    main()
