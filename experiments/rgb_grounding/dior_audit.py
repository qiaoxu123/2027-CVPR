"""Read actual DIOR-RSVG archives and count annotations; do not train GPS-unknown data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True);args=p.parse_args();root=Path(args.assets)/'dior_mirror';output=Path(args.output)
    if output.exists():raise ValueError('refusing archive audit overwrite')
    classes=Counter();descriptions=[];failures=Counter();sizes=Counter();objects=0
    with zipfile.ZipFile(root/'Annotations.zip') as archive:
        names=[n for n in archive.namelist() if n.endswith('.xml')]
        for name in names:
            doc=ET.fromstring(archive.read(name));w=int(doc.findtext('size/width'));h=int(doc.findtext('size/height'));sizes[f'{w}x{h}']+=1
            for obj in doc.findall('object'):
                objects+=1;classes[obj.findtext('name')]+=1;description=obj.findtext('description');descriptions.append(description)
                box=[float(obj.findtext('bndbox/'+key)) for key in ['xmin','ymin','xmax','ymax']]
                if not description or not description.strip():failures['empty_language']+=1
                if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):failures['bbox_outside_or_invalid']+=1
    with zipfile.ZipFile(root/'JPEGImages.zip') as archive:images=[n for n in archive.namelist() if n.lower().endswith(('.jpg','.jpeg','.png'))]
    splits={s:[int(v) for v in (root/(s+'.txt')).read_text().splitlines() if v.strip()] for s in ['train','val','test']}
    result=dict(actual_annotation_xml_files=len(names),actual_images=len(images),actual_object_expression_records=objects,unique_descriptions=len(set(descriptions)),object_classes=dict(classes),image_sizes=dict(sizes),failures=dict(failures),split_counts={s:len(v) for s,v in splits.items()},split_index_union_count=len(set().union(*map(set,splits.values()))),maximum_split_index=max(max(v) for v in splits.values()),split_pairwise_index_overlap={a+'_'+b:len(set(splits[a])&set(splits[b])) for a,b in [('train','val'),('train','test'),('val','test')]},archive_sha256={n:file_sha256(root/n) for n in ['Annotations.zip','JPEGImages.zip']},source='official RSVG repo -> third-party HF mirror; original official archive hash unavailable, mirror equivalence not certified',geographic_overlap='worldwide DIOR without per-sample GPS, unresolved for UK CityNav; not used for pretraining',training=False,metrics=False,split_semantics='index files appear to enumerate object/expression records, not image filenames; exact official reader ordering must be bound before training')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
