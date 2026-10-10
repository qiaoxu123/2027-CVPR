"""Group-clean AerialVG pilot; preserve official annotations and split audit."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def video_key(filename):
    parts=filename.split('_')
    if len(parts)<4:raise ValueError('unrecognized original VisDrone name')
    return parts[1]


def run(args):
    source=Path(args.assets)/'aerialvg';output=Path(args.output)
    if output.exists():raise ValueError('refusing external partition overwrite')
    output.mkdir(parents=True)
    rows=[json.loads(x) for x in (source/'annotation/vg_train_odvg.jsonl').read_text().splitlines()]
    validation=[json.loads(x) for x in (source/'annotation/vg_val_odvg.jsonl').read_text().splitlines()]
    train_names={r['filename'] for r in rows};val_names={r['filename'] for r in validation}
    groups=sorted({video_key(r['filename']) for r in rows})
    dev_groups={g for g in groups if int(hashlib.sha256(('externaldev:'+g).encode()).hexdigest(),16)%5==0}
    images=defaultdict(list)
    for row in rows:images[row['filename']].append(row)
    selected={};files=[];annotation_stats={}
    for split,limit in [('train',args.train_images),('internal_dev',args.dev_images)]:
        names=sorted((n for n in images if (video_key(n) in dev_groups)==(split=='internal_dev')),
                     key=lambda n:hashlib.sha256(('externalimage:'+n).encode()).hexdigest())[:limit]
        selected[split]=names;chosen=[]
        for name in names:
            # Dataset creation and duplicate audit, not outcome-dependent sampling.
            expressions=sorted(images[name],key=lambda r:hashlib.sha256(r['grounding']['caption'].encode()).hexdigest())[:4]
            chosen.extend(expressions)
            files.append(dict(name='aerialvg/images/'+name,url=f'https://huggingface.co/datasets/IPEC-COMMUNITY/AerialVG/resolve/{args.revision}/images/{name}',authenticated=True,revision=args.revision,limit_gb=.02))
        with (output/(split+'.jsonl')).open('w') as h:
            for r in chosen:h.write(json.dumps(r)+'\n')
        annotation_stats[split]=dict(images=len(names),expressions=len(chosen),source_video_groups=len({video_key(n) for n in names}))
    image_overlap=len(set(selected['train'])&set(selected['internal_dev']))
    group_overlap=len({video_key(n) for n in selected['train']}&{video_key(n) for n in selected['internal_dev']})
    if image_overlap or group_overlap:raise ValueError('external image/video leakage')
    audit=dict(official_train_expressions=len(rows),official_val_expressions=len(validation),official_train_unique_images=len(train_names),official_val_unique_images=len(val_names),official_train_val_image_overlap=len(train_names&val_names),
        official_val_unique_images_not_in_train=len(val_names-train_names),pilot=annotation_stats,pilot_image_overlap=image_overlap,pilot_source_video_overlap=group_overlap,
        selection='source-video SHA modulo5 heldout within official Train; fixed image/description hashes, not label/outcome selection',
        official_val_use='split audit only; not an image-independent development metric',
        full_caption_target_policy='all mentioned annotated regions are multipositive; relation-anchor boxes are not falsely forced into one primary target',
        phrase_policy='region phrase supervised with all same-normalized-phrase boxes in observed image; unannotated region relevance unknown',
        geographic_source='VisDrone-derived Chinese drone scenes; source-video grouping is conservative filename grouping, not supplied GPS proof')
    (output/'split_audit.json').write_text(json.dumps(audit,indent=2));(output/'image_download_spec.json').write_text(json.dumps(files,indent=2));print(json.dumps(audit),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True);p.add_argument('--revision',required=True);p.add_argument('--train-images',type=int,default=384);p.add_argument('--dev-images',type=int,default=96);run(p.parse_args())


if __name__=='__main__':main()
