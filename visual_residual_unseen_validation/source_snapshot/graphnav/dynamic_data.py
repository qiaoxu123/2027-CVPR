"""Label-separated causal visual states with exact frozen A3 goal scores."""
import json
from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from .candidate_geometry import FullCandidateDataset
from .checkpointing import file_sha256
from .trajectory_data import query_is_dev


class DynamicEvidenceDataset(Dataset):
    def __init__(self,evidence,queries,fields,vectors,args,*,history=False,split='all',split_seed=0,max_observed=32,associate_references=False):
        self.root=Path(evidence);self.vector_root=Path(vectors);self.field_root=Path(fields)
        rows=[json.loads(line) for line in (self.root/'samples.jsonl').read_text().splitlines() if line]
        self.static=FullCandidateDataset(queries,grid_size=args.grid_size,max_landmarks=args.max_landmarks,
            max_roads=args.max_roads,max_relations=args.max_relations,prior_sigma_m=args.prior_sigma_m,
            rule_sigma_m=args.rule_sigma_m,deduplicate_queries=False)
        self.rows=[row for row in rows if split=='all' or query_is_dev(self.static.records[row['query_index']],split_seed)==(split=='dev')]
        self.history,self.max_observed=history,max_observed
        evidence_manifest=json.loads((self.root/'manifest.json').read_text())
        if evidence_manifest['sha256']['queries']!=file_sha256(queries):raise ValueError('evidence/public query mismatch')
        if evidence_manifest['samples_sha256']!=file_sha256(self.root/'samples.jsonl'):raise ValueError('changed dynamic samples')
        vector_manifest=json.loads((self.vector_root/'manifest.json').read_text())
        if file_sha256(self.vector_root/'manifest.json')!=evidence_manifest['vector_manifest_sha256']:
            raise ValueError('changed visual vector protocol')
        if file_sha256(self.vector_root/'vectors.pt')!=vector_manifest['vectors_sha256']:raise ValueError('changed frozen visual vectors')
        self.vectors=torch.load(self.vector_root/'vectors.pt',map_location='cpu',weights_only=True)
        self.rois=[json.loads(line) for line in (self.vector_root/'rows.jsonl').read_text().splitlines() if line]
        if file_sha256(self.root/'geometry_quality.pt')!=evidence_manifest['geometry_sha256']:raise ValueError('changed geometric evidence')
        geometry=torch.load(self.root/'geometry_quality.pt',map_location='cpu',weights_only=True)
        self.geometry,self.confidence=geometry['geometry'],geometry['confidence']
        self.descriptors=geometry.get('entity_descriptors',{});self.associate_references=associate_references
        if associate_references and not self.descriptors:raise ValueError('public region descriptors required for visual node association')
        self._reference_matches={}
        self._descriptors_by_name=defaultdict(list)
        for key,descriptor in self.descriptors.items():
            self._descriptors_by_name[(key.rsplit(':',1)[0],descriptor['name'])].append((key,descriptor))
        manifest=json.loads((self.field_root/'manifest.json').read_text())
        if not manifest.get('complete') or manifest['protocol'].get('dense_logits_dtype')!='float32':
            raise ValueError('complete FP32 goal-score cache required; old rounded caches cannot be used')
        if manifest['protocol']['query_sha256']!=file_sha256(queries):raise ValueError('A3/public query SHA mismatch')
        # Keep only the query subset represented by these actual observations.
        needed={row['query_index'] for row in self.rows};self.fields={}
        for chunk in manifest['chunks']:
            indices=[i for i in needed if chunk['first_query']<=i<chunk['first_query']+chunk['queries']]
            if not indices:continue
            path=self.field_root/chunk['file']
            if file_sha256(path)!=chunk['sha256']:raise ValueError('changed A3 chunk')
            state=torch.load(path,map_location='cpu',weights_only=True)
            for index in indices:
                local=index-chunk['first_query']
                self.fields[index]=(state['belief_logits'][local].clone(),state['dense_context'][local].clone())
        if len(self.fields)!=len(needed):raise ValueError('missing frozen query fields')
        bert=torch.load(self.field_root/'bert_tokens.pt',map_location='cpu',weights_only=True)
        if bert['name']!=args.text_backbone or bert['max_length']!=getattr(args,'max_text_length',96):
            raise ValueError('BERT public-name encoding protocol mismatch')
        self.names={name:tokens[0].float().clone() for name,(tokens,_) in bert['cache'].items()};del bert

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];qid=row['query_index'];static=self.static[qid]
        # Labels in the public packing scaffold remain dummy and are not forwarded.
        observable={key:static[key] for key in ('nodes','node_types','node_valid','entity_valid','adjacency',
            'edges','pair_geometry','grid_xy')}
        try:observable['node_emb']=torch.stack([self.names[name] for name in static['node_names']])
        except KeyError as error:raise ValueError('missing frozen public-name embedding') from error
        logits,dense=self.fields[qid];observable.update(base_logits=logits,dense_context=dense)
        selected=row['history_roi_indices'] if self.history else row['current_roi_indices']
        groups={}
        for rid in selected:
            roi=self.rois[rid]
            if roi['source_episode_index']!=row['source_episode_index'] or roi['observed_step']>row['step']:
                raise ValueError('cross-episode/future visual evidence rejected')
            key=roi['map_name']+':'+roi['object_id'];weight=float(self.confidence[rid])
            if key not in groups:groups[key]=dict(vector=torch.zeros_like(self.vectors[rid],dtype=torch.float32),mass=0.,confidence=0.,pixels=0)
            group=groups[key];group['vector']+=self.vectors[rid].float()*weight;group['mass']+=weight
            group['confidence']=max(group['confidence'],weight);group['pixels']=max(group['pixels'],roi['visible_pixels'])
        ordered=sorted(groups,key=lambda key:(-groups[key]['confidence']*groups[key]['pixels'],key))[:self.max_observed]
        reference_vectors=torch.zeros(len(static['node_valid']),self.vectors.shape[-1]);reference_confidence=torch.zeros(len(static['node_valid']))
        if self.associate_references:
            if qid not in self._reference_matches:
                entities=self.static._prepared(qid)[0]['entities'];matches={}
                for node,entity in enumerate(entities):
                    poly=np.asarray(entity['geometry_m'])
                    if len(poly)>1 and np.linalg.norm(poly[-1]-poly[0])<1e-4:poly=poly[:-1]
                    for key,descriptor in self._descriptors_by_name[(row['map_name'],entity['name'])]:
                        source=descriptor['contour_world'].numpy()
                        if len(source)>1 and np.linalg.norm(source[-1]-source[0])<1e-4:source=source[:-1]
                        if source.shape==poly.shape and np.allclose(source,poly,atol=.02,rtol=0):matches[key]=node
                self._reference_matches[qid]=matches
            matches=self._reference_matches[qid];unreferenced=[]
            for key in ordered:
                group=groups[key]
                if key in matches:
                    node=matches[key];reference_vectors[node]=group['vector']/max(group['mass'],1e-8);reference_confidence[node]=group['confidence']
                else:unreferenced.append(key)
            ordered=unreferenced
        n=self.max_observed;q=len(static['grid_xy']);dim=self.vectors.shape[-1]
        vectors=torch.zeros(n,dim);confidence=torch.zeros(n);geometry=torch.zeros(q,n,10,dtype=torch.float16)
        for position,key in enumerate(ordered):
            group=groups[key];vectors[position]=group['vector']/max(group['mass'],1e-8)
            confidence[position]=group['confidence'];geometry[:,position]=self.geometry[key]
        observable.update(observed_vectors=vectors,observed_confidence=confidence,observed_geometry=geometry)
        if self.associate_references:observable.update(visual_vectors=reference_vectors,visual_confidence=reference_confidence)
        bounds=torch.tensor(self.static.records[qid]['world_bounds'],dtype=torch.float32)
        target=torch.tensor(row['labels']['target_world'][:2],dtype=torch.float32)
        labels=dict(target_xy=(target-bounds[:2])/(bounds[2:]-bounds[:2]),map_sizes=bounds[2:]-bounds[:2],
            query_index=torch.tensor(qid),source_step=torch.tensor(row['step']))
        return observable,labels
