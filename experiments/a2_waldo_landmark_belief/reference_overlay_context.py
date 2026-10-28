"""Bound TRAIN instruction/public-reference context and feature-cache signature."""
import json
from functools import lru_cache
from pathlib import Path

from .paths import DATA, ROOT, SOURCE, BERT_REVISION, sha
from .public_road_references import resolve_road_references


BASE = ROOT/'runs/a2_waldo_landmark_20261027'
DEFAULT_OUTPUT = BASE/'revision_v4/reference_cache'
CODE_FILES = [
    'core.py','data_protocol.py','mentions.py','paths.py','relation_v3_data.py',
    'public_regions.py','public_road_references.py','occurrence_mentions.py','reference_geometry.py',
    'reference_rebuild.py','reference_training.py','reference_overlay.py','reference_overlay_context.py',
    'reference_overlay_build.py','reference_overlay_data.py',
]


def feature_signature():
    """Snapshot actual extraction inputs; changing them invalidates continuation."""
    from huggingface_hub.constants import HF_HUB_CACHE
    source = json.loads((BASE/'source_manifest.json').read_text())
    snapshot = Path(HF_HUB_CACHE)/'models--bert-base-uncased/snapshots'/BERT_REVISION
    weights = snapshot/'model.safetensors'
    if not weights.is_file():
        raise ValueError('Exact frozen BERT safetensors snapshot required')
    data = dict(roles=BASE/'revision_v3/roles.json',source=BASE/'source_manifest.json',
                annotation=Path(source['annotation_path']),objects=DATA/'cityrefer/objects.json',
                descriptions=DATA/'cityrefer/processed_descriptions.json',
                v1_population=BASE/'revision_v1/confirmation_population.json',
                v2_population=BASE/'revision_v2/confirmation_population.json')
    files = {name:sha(path) for name,path in data.items()}
    if files['annotation'] != source['annotation_sha256']:
        raise ValueError('Frozen instruction annotation changed')
    return dict(kind='a2-reference-cache-signature-v1',bert_revision=BERT_REVISION,seed=0,
        train_observations=33440,dev_observations=5647,source_sha256=files,
        code_sha256={name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name) for name in CODE_FILES},
        bert_encoder_code_sha256=sha(SOURCE/'graphnav/token_language.py'),
        bert_snapshot_sha256={p.name:sha(p) for p in sorted(snapshot.iterdir()) if p.is_file()})


class TrainingReferenceContext:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        source = json.loads((BASE/'source_manifest.json').read_text())
        if sha(source['annotation_path']) != source['annotation_sha256']:
            raise ValueError('Frozen instruction annotation changed')
        self.records = json.loads(Path(source['annotation_path']).read_text())
        objects = json.loads((DATA/'cityrefer/objects.json').read_text())
        self.named = {m:[o for o in values.values() if o.get('name')] for m,values in objects.items()}
        self.processed = json.loads((DATA/'cityrefer/processed_descriptions.json').read_text())
        self.populations = {'teacher_snapshot':{e['index']:e for e in source['entries']}}
        for version in ('revision_v1','revision_v2'):
            pop = json.loads((BASE/version/'confirmation_population.json').read_text())
            self.populations[version+'_A0_closedloop'] = {e['index']:e for e in pop['entries']}

    @lru_cache(maxsize=20000)
    def _resolve(self, source_kind, episode, map_name, object_id):
        identity = self.populations[source_kind][episode]
        record = self.records[episode]
        expected = (map_name,object_id,identity['description_id'])
        actual = (f"{record['area']}_block_{record['block']}",int(record['object_ids'][0]),int(record['ann_ids'][0]))
        if actual != expected or (identity['map_name'],identity['object_id']) != (map_name,object_id):
            raise ValueError('Public instruction source identity changed')
        text = str(record['descriptions'][0])
        queries = list(dict.fromkeys(str(x) for x in self.processed[map_name][str(object_id)][identity['description_id']].get('landmarks',[])))
        if len(queries) != identity['reference_queries']:
            raise ValueError('Audited public reference query count changed')
        references = resolve_road_references(map_name,queries,self.named[map_name])
        offsets = self.tokenizer(text,truncation=True,max_length=96,return_offsets_mapping=True)['offset_mapping']
        return text,offsets,references

    def resolve(self, entry):
        if entry['role'] not in ('train','dev'):
            raise ValueError('Only authorized TRAIN/DEV references may be rebuilt')
        return self._resolve(entry['source_kind'],entry['episode'],entry['map_name'],entry['object_id'])
