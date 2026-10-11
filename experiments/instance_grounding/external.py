"""DIOR official XML ordering audit + frozen oracle-proposal diagnostic.

No transfer training: worldwide image geography remains unresolved.
"""
import argparse,hashlib,json,time,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
import cv2,numpy as np,torch
from experiments.rgb_grounding.city_features import CropEncoder
from graphnav.checkpointing import file_sha256
from .data import write


def run(a):
    out=Path(a.output)
    if out.exists():raise ValueError('no overwrite')
    out.mkdir(parents=True);start=time.time();assets=Path(a.assets)/'dior_mirror';cv2.setNumThreads(1);encoder=CropEncoder();rows=[];index=0;partition={s:set(map(int,(assets/(s+'.txt')).read_text().split())) for s in ['train','val','test']};official_images={s:set() for s in partition};entries=[]
    with zipfile.ZipFile(assets/'Annotations.zip') as z:
        for name in sorted(n for n in z.namelist() if n.endswith('.xml')):
            doc=ET.fromstring(z.read(name));filename=doc.findtext('filename');objs=[]
            for member in doc.findall('object'):
                box=[int(member.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']]
                objs.append(dict(box=box,phrase=member.findtext('description'),typ=member.findtext('name'),expression_index=index))
                for s,ids in partition.items():
                    if index in ids:official_images[s].add(filename)
                index+=1
            if len(objs)>=2:entries.append((filename,objs))
    chosen=sorted(entries,key=lambda e:hashlib.sha256(('dior-p2:'+e[0]).encode()).hexdigest())[:a.limit]
    with zipfile.ZipFile(assets/'JPEGImages.zip') as z:
        lookup={Path(n).name:n for n in z.namelist() if n.lower().endswith('.jpg')}
        for i,(filename,objects) in enumerate(chosen):
            image=z.read(lookup[filename]);bgr=cv2.imdecode(np.frombuffer(image,np.uint8),cv2.IMREAD_COLOR);objects=[o for o in objects if min(o['box'][2]-o['box'][0],o['box'][3]-o['box'][1])>=8][:32]
            if len(objects)<2:continue
            crops=[bgr[b:d,a:c] for o in objects for a,b,c,d in [o['box']]];vectors,_=encoder.encode(crops)
            for target in objects[:3]:
                phrase=target['phrase'];text=encoder.native.texts([phrase]).cpu().float()[0];scores=vectors.float()@text;rank=scores.argsort(descending=True).tolist();positives=[j for j,o in enumerate(objects) if o['phrase'].lower().strip()==phrase.lower().strip()]
                sameclass=[j for j,o in enumerate(objects) if o['typ']==target['typ']];hard_rank=sorted(sameclass,key=lambda j:-float(scores[j]));identity=objects.index(target)
                rows.append(dict(sample_id=filename+':'+str(target['expression_index']),family_id=filename,map_name='dior_geographic_unknown',instruction=phrase,target_index=identity,positive_indices=positives,scores=scores.tolist(),candidate_boxes=[o['box'] for o in objects],metrics=dict(attribute_phrase_r1=float(rank[0] in positives),identity_r1=float(rank[0]==identity),same_class_r1=float(hard_rank[0]==identity) if len(sameclass)>1 else None),image_sha256=hashlib.sha256(image).hexdigest()))
            if i%25==0:print(json.dumps(dict(images=i+1,queries=len(rows),seconds=time.time()-start)),flush=True)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    from experiments.bidirectional_core.stats import summarize_rows
    write(out/'metrics.json',dict(models={k:summarize_rows(rows,k,seed=20261011) for k in ['attribute_phrase_r1','identity_r1','same_class_r1']},images=len({r['family_id'] for r in rows}),queries=len(rows),seconds=time.time()-start,official_reader='sorted XML paths, member order, zero-based expression counter, official data_loader.py lines43-55',official_expression_count=index,official_image_split_overlap={a+'_'+b:len(official_images[a]&official_images[b]) for a,b in [('train','val'),('train','test'),('val','test')]},source_hashes={f:file_sha256(assets/f) for f in ['Annotations.zip','JPEGImages.zip']},mode='frozen SigLIP2 conditional on annotated oracle bbox proposals; not end-to-end bbox grounding, no CityNav transfer and no official DIOR test claim',geographic_overlap='unresolved worldwide source; quarantine',license='official RSVG CC BY-NC4.0, mirror not original hash-certified',peak_gpu_bytes=torch.cuda.max_memory_allocated()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=96);run(p.parse_args())
if __name__=='__main__':main()
