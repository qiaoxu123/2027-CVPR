"""Transfer navigation for matched map geometry arms; original head unchanged."""
import json, hashlib
import torch
from torch.nn import functional as F
from graphnav.geombelief_inputs import ARMS,SPATIAL,mask_geometry
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.candidate_geometry import collate_full
from graphnav.full_experiment import _batch_inputs
from scripts import polygon_ablation_navigation as nav
from scripts import polygon_ablation_engine as engine
from scripts.polygon_ablation_plan import canonical_job,query_key
from scripts.geombelief_map_experiments import OLD,OUT

class MapPolicy(nav.PolygonPolicy):
    donor_lookup=None
    donor_tensors=None
    @torch.no_grad()
    def frozen_field(self):
        key=json.dumps(self.record,sort_keys=True)
        if key not in self.fields:
            base,graph=nav.public_query_features(key,file_sha256(OUT/'frozen_plan.json'))
            item=nav.geometry_view(base,graph,self.record['world_bounds'],representation='full_polygon',scope='descriptor_only')
            x,t,m,n=_batch_inputs(collate_full([item]),self.encoder,self.device,'geometry')
            if self.representation=='shuffled':
                donor=self.donor_lookup[query_key(self.record)]
                for k in ('nodes','prior','prior_buildings','rules','relation_fields'):
                    x[k]=self.donor_tensors[k][donor:donor+1].cuda()
                p=self.donor_tensors['pair_geometry'][donor:donor+1].cuda()
                x['pair_geometry']=F.pad(p,(0,0,0,x['nodes'].shape[1]-p.shape[-2]))
            else:x=mask_geometry(x,self.representation)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                field=self.model(x,variant='geometry',tokens=t,token_mask=m,node_emb=n)
            field.dense_context=None;self.fields[key]=field
        return self.fields[key],None

def run_all(plan):
    nav.OUT=OUT;engine.OUT=OUT;nav.PolygonPolicy=MapPolicy
    encoder=engine.encoder_for(plan)
    for split in ('val_seen','val_unseen'):
        manifest=json.loads((OUT/f'{split}_shuffle_manifest.json').read_text())
        MapPolicy.donor_lookup={r['query_id']:r['donor_index'] for r in manifest}
        payload=torch.load(OLD/f'.cache/{split}_geometry.pt',map_location='cpu',weights_only=True,mmap=True)
        MapPolicy.donor_tensors=payload['tensors']
        for seed in (0,1,2):
            for arm in ARMS:
                job=canonical_job('map_geometry',arm,seed);cp=OUT/'training'/job/'best.pt'
                nav.navigate(plan,encoder,job,'map_geometry',arm,cp,split=split,smoke=True)
                nav.navigate(plan,encoder,job,'map_geometry',arm,cp,split=split)
    atomic_json(dict(status='Done',new_full_split_evaluations=30,correct_geometry_reused=6),OUT/'map_navigation_complete.json')
