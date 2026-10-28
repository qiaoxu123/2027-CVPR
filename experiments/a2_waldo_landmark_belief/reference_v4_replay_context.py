"""Public instruction/reference context restricted to fixed sealed VERIFY queries."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

from .paths import DATA,sha
from .reference_overlay_context import BASE
from .public_road_references import resolve_road_references
from .calibration_protocol import calibration_fold


def public_context_signature():
    path=BASE/'source_manifest.json';source=json.loads(path.read_text())
    annotation=Path(source['annotation_path'])
    if sha(annotation)!=source['annotation_sha256']:
        raise ValueError('Frozen original TRAIN annotation changed')
    return dict(source_manifest_sha256=sha(path),annotation_sha256=source['annotation_sha256'],
        objects_sha256=sha(DATA/'cityrefer/objects.json'),
        descriptions_sha256=sha(DATA/'cityrefer/processed_descriptions.json'))


class ConfirmationReferenceContext:
    def __init__(self,tokenizer,entries):
        self.entries={entry['index']:deepcopy(entry) for entry in entries}
        if (not entries or len(self.entries)!=len(entries)
            or any(e.get('fold')!='verify' or calibration_fold(e['map_name'],e['object_id'])!='verify' for e in entries)):
            raise ValueError('Distinct predeclared sealed VERIFY queries required')
        self.signature=public_context_signature();self.tokenizer=tokenizer
        source=json.loads((BASE/'source_manifest.json').read_text())
        by_index={e['index']:e for e in source['entries']}
        if len(by_index)!=len(source['entries']):
            raise ValueError('Original query identity is not unique')
        for index,entry in self.entries.items():
            original=by_index.get(index)
            if original is None or any(entry.get(key)!=value for key,value in original.items()):
                raise ValueError('Sealed public query differs from original instruction source')
        self.records=json.loads(Path(source['annotation_path']).read_text())
        objects=json.loads((DATA/'cityrefer/objects.json').read_text())
        self.named={name:[item for item in items.values() if item.get('name')] for name,items in objects.items()}
        self.descriptions=json.loads((DATA/'cityrefer/processed_descriptions.json').read_text())

    @lru_cache(maxsize=130)
    def _resolve(self,index):
        query=self.entries[index];record=self.records[index]
        identity=(f"{record['area']}_block_{record['block']}",int(record['object_ids'][0]),int(record['ann_ids'][0]))
        if identity!=(query['map_name'],query['object_id'],query['description_id']):
            raise ValueError('Sealed instruction identity differs from original annotation')
        instruction=str(record['descriptions'][0])
        queries=list(dict.fromkeys(str(name) for name in
            self.descriptions[query['map_name']][str(query['object_id'])][query['description_id']].get('landmarks',[])))
        if len(queries)!=query['reference_queries']:
            raise ValueError('Original public reference query count changed')
        references=resolve_road_references(query['map_name'],queries,self.named[query['map_name']])
        offsets=self.tokenizer(instruction,truncation=True,max_length=96,return_offsets_mapping=True)['offset_mapping']
        return instruction,offsets,references

    def resolve(self,query):
        if query.get('index') not in self.entries or query!=self.entries[query['index']]:
            raise ValueError('Only the exact fixed sealed public query may be resolved')
        return deepcopy(self._resolve(query['index']))
