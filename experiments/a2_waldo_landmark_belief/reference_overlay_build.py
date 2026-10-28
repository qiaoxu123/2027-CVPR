"""CPU-only complete TRAIN/DEV reference rebuilding with immutable parents."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
from pathlib import Path
import sys
import time

import torch

from .paths import SOURCE, BERT_REVISION, save, sha
from .reference_overlay_context import DEFAULT_OUTPUT, feature_signature, TrainingReferenceContext
from .reference_overlay import (overlay_filename, make_reference_overlay, apply_reference_overlay,
                                verify_overlay_population)
from .reference_rebuild import rebuild_reference_pack
from .reference_training import build_reference_training_row
from .relation_v3_data import RelationV3Views, read_training_source, build_training_row


def build(directory=DEFAULT_OUTPUT):
    if torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
        raise ValueError('Run CPU-only with CUDA_VISIBLE_DEVICES empty; no parallel GPU extraction')
    torch.set_num_threads(2)
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'builder.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _build_locked(directory)


def _build_locked(directory):
    started = time.perf_counter()
    signature = feature_signature()
    signature_file = directory/'signature.json'
    if signature_file.exists():
        if json.loads(signature_file.read_text()) != signature:
            raise ValueError('Existing feature cache signature differs; preserve it and choose a new directory')
    else:
        if any(directory.iterdir()) and {p.name for p in directory.iterdir()} != {'builder.lock'}:
            raise ValueError('Unbound nonempty feature output directory')
        save(signature,signature_file)
    signature_sha = sha(signature_file)
    save(dict(state='preparing',phase='load_frozen_encoder_and_checked_train_dev',processed=0,total=39087),directory/'status.json')
    views = directory/'views'; views.mkdir(exist_ok=True)
    # Appending GraphNav avoids shadowing this project's scripts package.
    sys.path.append(str(SOURCE))
    from graphnav.token_language import FrozenBERTTokens
    bert = FrozenBERTTokens(device='cpu')
    if bert.revision != BERT_REVISION or any(p.requires_grad for p in bert.model.parameters()):
        raise ValueError('Frozen BERT protocol differs')
    context = TrainingReferenceContext(bert.tokenizer)
    datasets = {role:RelationV3Views(role) for role in ('train','dev')}
    records,counts,resumed = [],defaultdict(Counter),0
    expected = [entry for dataset in datasets.values() for entry in dataset.entries]
    if len(expected) != 39087:
        raise ValueError('Complete TRAIN/DEV population required')
    for role,dataset in datasets.items():
        for entry in dataset.entries:
            raw = read_training_source(entry,role,dataset.allowed,dataset.contexts[entry['source_kind']]['signature'])
            output = views/overlay_filename(entry)
            if output.exists():
                overlay = torch.load(output,map_location='cpu',weights_only=True)
                rebuilt = apply_reference_overlay(raw,overlay,entry,signature_sha)
                resumed += 1
            else:
                instruction,offsets,references = context.resolve(entry)
                if 'instruction' in raw and raw['instruction'] != instruction:
                    raise ValueError('Original cached instruction differs from bound annotation')
                public,meta = rebuild_reference_pack(raw['public'],raw['meta'],instruction=instruction,
                    offsets=offsets,references=references,encode_names=bert.encode_tokens)
                overlay = make_reference_overlay(public,meta,entry,signature_sha)
                rebuilt = apply_reference_overlay(raw,overlay,entry,signature_sha)
                temporary = output.with_suffix('.tmp')
                torch.save(overlay,temporary)
                temporary.replace(output)
                # Exercise the actual safe loading path, not just an in-memory
                # object that might contain unsupported pickle metadata.
                overlay = torch.load(output,map_location='cpu',weights_only=True)
                rebuilt = apply_reference_overlay(raw,overlay,entry,signature_sha)
            target_kind = dataset.target_kinds[(entry['map_name'],entry['object_id'])]
            real = entry['source_kind'] != 'teacher_snapshot'
            old_labels = build_training_row(raw,target_kind=target_kind,real=real)['labels']
            new_labels = build_reference_training_row(rebuilt,target_kind=target_kind,real=real)['labels']
            before,after = raw['meta']['valid_references'],rebuilt['meta']['valid_references']
            old_coverage,new_coverage = bool(old_labels['localization_utility'].any()),bool(new_labels['localization_utility'].any())
            for bucket in (role,role+'/'+entry['source_kind']):
                counts[bucket].update(observations=1,old_any_reference=int(before>0),new_any_reference=int(after>0),
                    recovered_reference=int(before==0 and after>0),lost_reference=int(before>0 and after==0),
                    old_spatially_covered=int(old_coverage),new_spatially_covered=int(new_coverage),
                    recovered_spatial_coverage=int(not old_coverage and new_coverage),
                    lost_spatial_coverage=int(old_coverage and not new_coverage))
            record = dict(entry,overlay_sha256=sha(output))
            records.append(record)
            if len(records)%100 == 0:
                status = dict(state='building',processed=len(records),total=len(expected),resumed=resumed,
                              seconds=time.perf_counter()-started,counts={k:dict(v) for k,v in counts.items()})
                save(status,directory/'status.json')
                print(json.dumps({k:status[k] for k in ('state','processed','total','resumed','seconds')}),flush=True)
    verify_overlay_population(records,expected)
    for entry in records:
        if sha(Path(entry['directory'])/entry['filename']) != entry['sha256'] or sha(views/overlay_filename(entry)) != entry['overlay_sha256']:
            raise ValueError('Parent or rebuilt feature bytes changed during extraction')
    if feature_signature() != signature or torch.cuda.is_initialized():
        raise ValueError('Feature source changed during extraction or CUDA was opened')
    manifest = dict(status='all_train_dev_reference_overlays_complete',signature_sha256=signature_sha,
        entries=records,counts={k:dict(v) for k,v in counts.items()},parent_cache_unchanged=True,
        no_gpu_used=True,calibration_tensors_opened=False,benchmark_used=False,
        limitation='Reference/candidate availability only, not navigation or new model performance.')
    save(manifest,directory/'manifest.json')
    save(dict(state='complete',processed=len(records),total=len(expected),resumed=resumed,
              seconds=time.perf_counter()-started,manifest_sha256=sha(directory/'manifest.json')),
         directory/'status.json')
    print(json.dumps(dict(state='complete',observations=len(records),counts=manifest['counts'])),flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=DEFAULT_OUTPUT)
    build(parser.parse_args().directory)


if __name__ == '__main__':
    main()
