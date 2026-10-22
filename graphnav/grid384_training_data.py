"""Real 384 candidate features from original causal poses; no resizing old ROI arrays."""
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
import hashlib,json,shutil,sys
import cv2,numpy as np,torch
from torch.nn import functional as F
from .stage1_data import Stage1Dataset,CandidateMemory
from .candidate_geometry import FullCandidateDataset,collate_full
from .appearance_matching import ObservedSigLIPMatchingEncoder
from .observed_patch_scales import observed_region_masks,masked_native_pool
from .checkpointing import atomic_save,file_sha256

class Grid384Dataset(Stage1Dataset):
    def __init__(self,root,verify=True):
        super().__init__(root,verify=verify)
        args=dict(self.manifest['protocol']['a3_args']);args['grid_size']=384
        self.static=FullCandidateDataset(self.root/'queries.jsonl',**{k:args[k] for k in ('grid_size','max_landmarks','max_roads','max_relations','prior_sigma_m','rule_sigma_m')},deduplicate_queries=False)
        self.static._prepared=lru_cache(maxsize=2)(self.static._prepare)
        self.static._pair_cache=lru_cache(maxsize=2)(self.static._pair_features)
        self.manifest={**self.manifest,'protocol':{**self.manifest['protocol'],'a3_args':args}}

class OnlineCollate:
    def __init__(self,source,data_root,cache,revision):
        sys.path.insert(0,source)
        from sbfnav.mapdata import GROUND_LEVEL
        from .renderer import CityNavObservedViewRenderer
        self.ground=GROUND_LEVEL
        self.renderer=CityNavObservedViewRenderer(Path(data_root)/'rgbd',Path(cache)/'unused',image_size=384)
        self.native=ObservedSigLIPMatchingEncoder(microbatch=1)
        assert getattr(self.native.model.config,'_commit_hash',None)==revision
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.memory=OrderedDict();self.bytes=sum(p.stat().st_size for p in self.cache.glob('*.pt'))
        self.limit=160*2**30
        self.geometry_sha=file_sha256(Path(__file__).with_name('observed_patch_scales.py'))
    def frame(self,map_name,world,old):
        pose=old['pose'].numpy();key=hashlib.sha256(map_name.encode()+pose.tobytes()+world.tobytes()+self.geometry_sha.encode()).hexdigest()
        path=self.cache/(key+'.pt')
        if key in self.memory:
            self.memory.move_to_end(key);return dict(self.memory[key],step=int(old["step"]))
        if path.exists():
            f=torch.load(path,map_location='cpu',weights_only=True)
        else:
            if pose[2]-self.ground[map_name]<=0:
                valid=np.zeros((len(world),3),bool);quality=np.zeros((len(world),3,3),np.float32);ids=np.empty(0,np.int64);vectors=torch.empty(0,1024,dtype=torch.float16);inverse=np.empty(0,np.int64)
            else:
                observation=self.renderer.current_rgbd(map_name,pose,self.ground[map_name])
                masks,quality=observed_region_masks(observation,world)
                valid=masks.any(-1);ids=np.flatnonzero(valid.any(-1))
                packed=np.packbits(masks[valid],axis=-1)
                unique,inverse=np.unique(packed,axis=0,return_inverse=True)
                del masks,packed
                if len(unique):
                    batch=self.native.processor(images=[cv2.cvtColor(observation['bgr'],cv2.COLOR_BGR2RGB)],return_tensors='pt').to('cuda')
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        patches=self.native.model.vision_model(**batch).last_hidden_state[0]
                        region=torch.from_numpy(np.unpackbits(unique,axis=-1,count=576).astype(bool)).cuda()
                        vectors=F.normalize(masked_native_pool(self.native.model.vision_model.head,patches,region).float(),dim=-1).cpu().half()
                else:vectors=torch.empty(0,1024,dtype=torch.float16)
            f=dict(roi_ids=torch.from_numpy(ids),roi_valid=torch.from_numpy(valid[ids]),roi_quality=torch.from_numpy(quality[ids]),vectors=vectors,inverse=torch.from_numpy(inverse.astype(np.int64)),pose=old['pose'],step=int(old['step']),grid_size=384)
            # Cache only newly generated public frozen features; never reuse old 28 ROI values.
            if self.bytes<self.limit and shutil.disk_usage(self.cache).free>40*2**30:
                atomic_save(f,path);self.bytes+=path.stat().st_size
        self.memory[key]=f
        while len(self.memory)>32:self.memory.popitem(last=False)
        return dict(f,step=int(old["step"]))
    def expand(self,f):
        valid=f['roi_valid'].cuda();v=torch.zeros(*valid.shape,1024,device='cuda',dtype=torch.float16)
        if valid.any():v[valid]=f['vectors'].cuda()[f['inverse'].cuda()]
        return dict(roi_ids=f['roi_ids'].cuda(),roi_features=v,roi_valid=valid,roi_quality=f['roi_quality'].cuda(),pose=f['pose'],step=f['step'])
    def __call__(self,items):
        if len(items)>1:
            batches=[self([item]) for item in items]
            return collate_full([item[0] for item in items]),{k:torch.cat([b[1][k] for b in batches]) for k in batches[0][1]},{k:torch.cat([b[2][k] for b in batches]) for k in batches[0][2]}
        static_item,row,frame,past=items[0];static=collate_full([static_item]);q=static['grid_xy'].shape[1];assert q==147456
        bounds=np.array(static['world_bounds'][0],np.float32);world=bounds[:2]+static['grid_xy'][0].numpy()*(bounds[2:]-bounds[:2])
        # Source query map is not inferred from GT.
        map_name=self.map_for_query[int(row['query_index'])]
        memory=CandidateMemory(q,1024,'cuda')
        for old in past:memory.update(self.expand(self.frame(map_name,world,old)))
        history,history_valid,history_quality=memory.tensors();del memory
        current=self.expand(self.frame(map_name,world,frame))
        images=torch.zeros(q,3,1024,device='cuda',dtype=torch.float16);valid=torch.zeros(q,3,device='cuda',dtype=torch.bool);quality=torch.zeros(q,3,3,device='cuda')
        ids=current['roi_ids'];images[ids]=current['roi_features'];valid[ids]=current['roi_valid'];quality[ids]=current['roi_quality']
        public=dict(native_text=row['native_text'][None].float(),appearance_text=row['appearance_text'][None].float(),roi_features=images[None],roi_valid=valid[None],roi_quality=quality[None],history_features=history[None],history_valid=history_valid[None],history_quality=history_quality[None],pose_world=row['pose_world'][None],bounds=torch.tensor(bounds)[None],ground_level=row['ground_level'][None])
        labels={k:row['labels'][k][None] for k in ('target_world','teacher_world','valid_teacher_z')}
        return static,public,labels
