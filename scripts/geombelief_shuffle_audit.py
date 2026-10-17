"""Count effective geometry corruption, including duplicate public map bundles."""
import json,time
import torch
from graphnav.checkpointing import atomic_json
from scripts.geombelief_map_experiments import OUT,OLD

def main():
    torch.set_num_threads(1);result=[]
    for split in ('train_seen','val_seen','val_unseen'):
        path=OUT/f'{split}_shuffle_manifest.json'
        while not path.exists():time.sleep(30)
        manifest=json.loads(path.read_text());p=torch.load(OLD/f'.cache/{split}_geometry.pt',map_location='cpu',weights_only=True,mmap=True)
        donors=torch.tensor([r['donor_index'] for r in manifest]);n=len(donors);changed=torch.zeros(n,dtype=torch.bool);counts={}
        for key in ('nodes','pair_geometry','prior','prior_buildings','rules','relation_fields'):
            t=p['tensors'][key];different=torch.zeros(n,dtype=torch.bool)
            for start in range(0,n,64):
                end=min(n,start+64);different[start:end]=(t[start:end]!=t[donors[start:end]]).flatten(1).any(1)
            changed|=different;counts[key]=int(different.sum())
        result.append(dict(split=split,queries=n,singleton_schema_queries=sum(r['unchanged'] for r in manifest),effective_changed_queries=int(changed.sum()),effective_unchanged_queries=int((~changed).sum()),per_path_changed=counts,
            limitation='Non-singleton queries can share identical spatial bundles; count these as uncorrupted rather than claiming every permutation destroys grounding. This weakens, not strengthens, the intervention.'))
    atomic_json(result,OUT/'shuffle_effectiveness.json')
if __name__=='__main__':main()
