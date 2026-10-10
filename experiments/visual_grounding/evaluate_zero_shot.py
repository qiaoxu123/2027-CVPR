"""Run real pinned models on the same frozen observed-only benchmark.

Weak language labels and image-review diagnostics are explicitly separated.
This runner refuses to edit any frozen dataset or existing completed evaluation.
"""
import argparse, json, time, importlib, sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from experiments.visual_grounding.freeze_dataset import read,sha
from experiments.visual_grounding.statistics import intervals,macro_f1

LABELS={'roof_color':['white','gray','red','blue','brown','black','green','yellow','orange','beige'],
        'building_color':['white','gray','red','blue','brown','black','green','yellow','orange','beige'],
        'vehicle_color':['white','gray','red','blue','brown','black','green','yellow','orange','beige'],
        'building_shape':['rectangular','square','round','circular','l-shaped','u-shaped']}


def prompt(kind,label):
    if kind=='roof_color':return f'a building with a {label} roof'
    if kind=='building_color':return f'a {label} building'
    if kind=='vehicle_color':return f'a {label} vehicle'
    if kind=='building_shape':return f'a {label} building'
    raise ValueError(kind)


def tensor(x):return torch.as_tensor(x,dtype=torch.float32).cpu()


def load_encoder(a):
    if a.model=='siglip2':
        from experiments.visual_grounding.adapters.siglip2 import Encoder
        return Encoder(device=a.device,microbatch=a.batch_size)
    if a.model.startswith('remoteclip'):
        from experiments.visual_grounding.adapters.remoteclip import Encoder
        return Encoder(model_name='ViT-L-14' if a.model.endswith('l14') else 'ViT-B-32',checkpoint=a.checkpoint,device=a.device,batch_size=a.batch_size)
    from experiments.visual_grounding.adapters.regionclip import RegionCLIPEncoder
    return RegionCLIPEncoder(checkpoint=a.checkpoint,official_repo=a.official_repo,device=a.device,batch_size=a.batch_size)


def validate(x):
    if x.ndim!=2 or not torch.isfinite(x).all():raise ValueError('nonfinite or invalid embeddings')
    if not torch.allclose(x.norm(dim=-1),torch.ones(len(x)),atol=.002):raise ValueError('embeddings must be normalized')


def encode(a,data,out,encoder):
    index=read(data/'index.jsonl');texts=sorted(set([q[k] for q in read(data/'queries.jsonl') for k in ['instruction','attribute_phrase'] if q[k].strip()] + [prompt(k,v) for k,vs in LABELS.items() for v in vs]))
    for name in ['index','queries','retrieval','attributes']:
        if sha(data/f'{name}.jsonl')!=json.loads((data/'frozen.json').read_text())['files'][name]:raise ValueError('frozen dataset changed')
    protocol=dict(frozen_sha256=sha(data/'frozen.json'),model=a.model,checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else None,region_source='source_frame_bbox' if a.model=='regionclip' else 'common_context025_letterbox_roi',batch_size=a.batch_size)
    if (out/'embeddings.pt').exists():
        state=torch.load(out/'embeddings.pt',map_location='cpu',weights_only=True)
        if state['protocol']!=protocol:raise ValueError('cache protocol mismatch')
        return index,state['images'],state['texts'],state['strings'],state['telemetry']
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats() if a.device=='cuda' else None
    # Smoke includes real source image and a real query; this is not a metric label.
    first=cv2.imread(index[0]['roi_path']);assert first is not None
    smoke=tensor(encoder.regions([cv2.imread(index[0]['roi_source'])],[index[0]['roi_bbox']]) if a.model=='regionclip' else encoder.images([first]));text=tensor(encoder.texts(['a building with a gray roof']));validate(smoke);validate(text)
    (out/'smoke_runner.json').write_text(json.dumps(dict(finite=True,image_shape=list(first.shape),image_embedding=list(smoke.shape),text_embedding=list(text.shape),sample_id=index[0]['sample_id']),indent=2))
    load_seconds=time.perf_counter()-start;vectors={};image_start=time.perf_counter();io_seconds=0
    if a.model=='regionclip':
        bysource=defaultdict(list)
        for i,r in enumerate(index):bysource[r['roi_source']].append((i,r))
        for counter,(path,rows) in enumerate(bysource.items()):
            t=time.perf_counter();image=cv2.imread(path);io_seconds+=time.perf_counter()-t
            if image is None:raise FileNotFoundError(path)
            # One C4 feature map shared by regions of this actually observed frame.
            for offset in range(0,len(rows),a.batch_size):
                sub=rows[offset:offset+a.batch_size];v=tensor(encoder.regions(image,[r['roi_bbox'] for _,r in sub]));validate(v)
                for (i,_),emb in zip(sub,v):vectors[i]=emb
            if counter%100==0:print(json.dumps(dict(encoded=len(vectors),total=len(index),seconds=time.perf_counter()-image_start)),flush=True)
        images=torch.stack([vectors[i] for i in range(len(index))])
    else:
        pieces=[]
        for offset in range(0,len(index),a.batch_size):
            t=time.perf_counter();batch=[cv2.imread(r['roi_path']) for r in index[offset:offset+a.batch_size]];io_seconds+=time.perf_counter()-t
            if any(x is None for x in batch):raise FileNotFoundError('ROI source')
            x=tensor(encoder.images(batch));validate(x);pieces.append(x)
            if offset%(a.batch_size*100)==0:print(json.dumps(dict(encoded=offset+len(batch),total=len(index),seconds=time.perf_counter()-image_start)),flush=True)
        images=torch.cat(pieces)
    torch.cuda.synchronize() if a.device=='cuda' else None
    image_seconds=time.perf_counter()-image_start;t=time.perf_counter();language=tensor(encoder.texts(texts));validate(language)
    torch.cuda.synchronize() if a.device=='cuda' else None
    telemetry=dict(images=len(images),image_seconds=image_seconds,roi_io_seconds=io_seconds,throughput_roi_s=len(images)/image_seconds,texts=len(texts),text_seconds=time.perf_counter()-t,smoke_seconds=load_seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if a.device=='cuda' else None,embedding_dimension=images.shape[-1],parameters=sum(p.numel() for p in encoder.model.parameters()) if hasattr(encoder,'model') else sum(p.numel() for p in encoder.visual.parameters())+sum(p.numel() for p in encoder.language.parameters()))
    torch.save(dict(protocol=protocol,images=images,texts=language,strings=texts,telemetry=telemetry),out/'embeddings.pt')
    metadata=encoder.provenance() if hasattr(encoder,'provenance') else encoder.metadata
    (out/'model_provenance_benchmark.json').write_text(json.dumps(metadata,indent=2,default=str))
    return index,images,language,texts,telemetry


def summarize(predictions):
    result={}
    for split in ['train_seen','val_seen','val_unseen']:
        rows=[r for r in predictions if r['split']==split];metrics={}
        for task in sorted(set(r['task'] for r in rows)):
            selected=[r for r in rows if r['task']==task];bycategory={}
            for category in sorted(set(r['category'] for r in selected)):
                sub=[r for r in selected if r['category']==category]
                bycategory[category]=intervals([r['correct'] for r in sub],[r['map_name'] for r in sub])
            m=intervals([r['correct'] for r in selected],[r['map_name'] for r in selected]);m['by_attribute_or_type']=bycategory
            if task.endswith('attribute'):
                m['macro_f1']=macro_f1([r['truth'] for r in selected],[r['prediction'] for r in selected])
            if selected and 'candidate_size' in selected[0]:
                m['candidate_size_mean']=float(np.mean([r['candidate_size'] for r in selected]));m['candidate_size_range']=[min(r['candidate_size'] for r in selected),max(r['candidate_size'] for r in selected)]
            objects=set((r['map_name'],r.get('object_id',r.get('query_id'))) for r in selected);m['independent_objects_or_queries']=len(objects)
            m['by_map']={name:dict(n=len(sub:=[r for r in selected if r['map_name']==name]),mean=float(np.mean([r['correct'] for r in sub]))) for name in sorted(set(r['map_name'] for r in selected))}
            metrics[task]=m
        result[split]=metrics
    return result


def evaluate(data,index,images,language,strings,split="development"):
    def allowed(row):return split=="all" or (row["split"]=="val_unseen")== (split=="unseen")
    lookup={r['sample_id']:i for i,r in enumerate(index)};textlookup={s:i for i,s in enumerate(strings)};pred=[]
    for row in read(data/'attributes.jsonl'):
        if not allowed(row):continue
        kind=row['attribute_type'];labels=LABELS.get(kind,[])
        if row['label'] not in labels:continue
        scores=language[[textlookup[prompt(kind,v)] for v in labels]]@images[lookup[row['sample_id']]]
        prediction=labels[int(scores.argmax())];verified=row['label_status']=='agent_image_review_exploratory'
        pred.append(dict(task='image_review_attribute' if verified else 'weak_attribute',category=kind,split=row['split'],map_name=row['map_name'],object_id=row['object_id'],sample_id=row['sample_id'],truth=row['label'],prediction=prediction,correct=prediction==row['label'],label_status=row['label_status'],scores=dict(zip(labels,scores.tolist()))))
    for q in read(data/'queries.jsonl'):
        if not allowed(q):continue
        ids=q['candidate_sample_ids'];positive=set(q['positive_sample_ids']);negative=set(q['negative_sample_ids']);x=images[[lookup[s] for s in ids]]
        for mode in ['instruction','attribute_phrase']:
            scores=x@language[textlookup[q[mode]]];order=scores.argsort(descending=True).tolist();ranking=[ids[i] for i in order]
            common=dict(category=q['attribute_type'],split=q['split'],map_name=q['map_name'],object_id=index[lookup[q['query_sample_id']]]['object_id'],query_id=q['query_sample_id'],candidate_size=len(ids),positive_count=len(positive),ranking=ranking,scores=dict(zip(ids,scores.tolist())),label_status=q['label_status'])
            pred.append(dict(common,task='weak_hard_negative_'+mode+'_r1',correct=ranking[0] in positive))
            pred.append(dict(common,task='weak_text_region_'+mode+'_r5',correct=bool(set(ranking[:5])&positive)))
            pi=[i for i,s in enumerate(ids) if s in positive];ni=[i for i,s in enumerate(ids) if s in negative]
            preference=float((scores[pi,None]>scores[ni][None]).float().mean())
            pred.append(dict(common,task='weak_pairwise_'+mode,correct=preference))
        # Attribute replacement only for supported colors/shapes within this same set.
        kind=q['attribute_type'];alternative_labels=sorted(set(index[lookup[s]]['attributes'][kind]['label'] for s in ids if s in negative))
        for label in alternative_labels:
            actual_positive={s for s in ids if index[lookup[s]]['attributes'][kind]['label']==label}
            scores=x@language[textlookup[prompt(kind,label)]];top=ids[int(scores.argmax())]
            original_scores=x@language[textlookup[prompt(kind,q['attribute_label'])]]
            pred.append(dict(task='weak_counterfactual',category=kind,split=q['split'],map_name=q['map_name'],object_id=index[lookup[q['query_sample_id']]]['object_id'],query_id=q['query_sample_id'],from_label=q['attribute_label'],to_label=label,correct=top in actual_positive,chosen=top,candidate_size=len(ids),mean_absolute_score_change=float((scores-original_scores).abs().mean()),label_status='language_weak_not_visual_verified'))
    gallery=[r for r in read(data/'retrieval.jsonl') if allowed(r)]
    groups=defaultdict(list)
    for r in gallery:groups[r['split'],r['map_name'],r['object_type']].append(r)
    for key,rows in sorted(groups.items()):
        if len(rows)<2:continue
        x=images[[lookup[r['query_sample_id']] for r in rows]];g=images[[lookup[r['gallery_sample_id']] for r in rows]];scores=x@g.T
        for i,r in enumerate(rows):
            order=scores[i].argsort(descending=True).tolist()
            for k in [1,5]:pred.append(dict(task=f'cross_view_instance_r{k}',category=key[2],split=key[0],map_name=key[1],object_id=r['object_id'],sample_id=r['query_sample_id'],gallery_sample_id=r['gallery_sample_id'],candidate_size=len(rows),correct=i in order[:k],ranking=[rows[j]['object_id'] for j in order[:5]],note='different actual camera footprints; gallery is cross-episode offline instance retrieval, not policy causal evidence'))
    return pred


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['remoteclip_l14','remoteclip_b32','siglip2','regionclip'],required=True);p.add_argument('--checkpoint');p.add_argument('--official-repo');p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--split',choices=['development','unseen','all'],default='development');a=p.parse_args()
    data=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);resultfile=out/f'metrics_{a.split}.json'
    if resultfile.exists():raise ValueError('completed evaluation exists; refusing repeated evaluation')
    # Encode all images only after dataset/config freeze; evaluate Unseen once separately.
    encoder=load_encoder(a);index,images,language,strings,telemetry=encode(a,data,out,encoder)
    predictions=evaluate(data,index,images,language,strings,a.split)
    predictions=[r for r in predictions if a.split=='all' or (r['split']=='val_unseen')==(a.split=='unseen')]
    (out/f'predictions_{a.split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    metrics=dict(model=a.model,frozen_sha256=sha(data/'frozen.json'),results=summarize(predictions),telemetry=telemetry,verified_semantic_benchmark=None,limitation='language weak labels and exploratory agent image review do not replace independent human attribute annotation',evaluation_split=a.split)
    resultfile.write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
