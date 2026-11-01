"""Collect native public evidence on actual Train rollouts, then attach labels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return torch.from_numpy(value.copy())
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    return value


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--native-checkpoint',required=True)
    a,rest=p.parse_known_args()
    def value(flag):return rest[rest.index(flag)+1]
    if value('--split')!='train_seen':raise ValueError('Collector is Train-only')
    output=Path(value('--output'))
    if output.exists():raise ValueError('Refusing overwrite')
    from graphnav import stage1_runtime
    from graphnav.native_object_runtime import install_native_object_belief
    original=stage1_runtime.GraphNavStage1Policy
    instances=[]
    class CollectPolicy(original):
        def __init__(self,checkpoint,directory,**kwargs):
            super().__init__(checkpoint,directory,**kwargs)
            install_native_object_belief(self,a.native_checkpoint,checkpoint)
            self.rows=[];self.files=[];instances.append(self)
            self.native_collection_sink=self.collect
        def flush(self):
            if not self.rows:return
            name=f'evidence_{len(self.files):04d}.pt'
            atomic_save(self.rows,output/name)
            self.files.append(dict(file=name,states=len(self.rows),sha256=file_sha256(output/name)))
            self.rows=[]
        def collect(self,row):
            if int(row['step'])==0:self.flush()
            self.rows.append(cpu(row))
    stage1_runtime.GraphNavStage1Policy=CollectPolicy
    sys.argv=[sys.argv[0]]+rest
    from scripts.evaluate_native_shared_closedloop import main as evaluate
    try:evaluate()
    finally:
        for policy in instances:policy.flush()
    episodes=json.loads((output/'episodes.json').read_text())
    files=instances[0].files
    if len(files)!=len(episodes):raise ValueError('Evidence/episode count mismatch')
    # Dataset private targets enter only here, after every rollout has finished.
    from sbfnav.dataset import CityNavDataset
    source=CityNavDataset(value('--data-root'),'train_seen',version='processed_citynav')
    labels=[]
    for e,f in zip(episodes,files):
        r=source[e['index']]
        if list(r.episode_id)!=e['episode_id']:raise ValueError('Episode identity mismatch')
        labels.append(dict(index=e['index'],episode_id=e['episode_id'],map_name=r.map_name,instruction=r.instruction,target_world=np.asarray(r.target_position).tolist(),**f))
    atomic_json(dict(complete=True,partition='train',kind='native-onpolicy-evidence-v1',episodes=labels,native_sha=file_sha256(a.native_checkpoint),base_sha=file_sha256(value('--checkpoint')),selection_manifest_sha=file_sha256(value('--train-collection-manifest')),protocol='Public evidence only during inference. Targets attached after complete rollouts; never consumed by policy.'),output/'collection.json')


if __name__=='__main__':main()
