"""Versioned TRAIN-only label audit; preserve public cached observations exactly."""
import copy
import json
import numpy as np
import torch
from .paths import bootstrap,ROOT,DATA,sha,save
bootstrap()
from .private_labels import PrivateAssociationIndex
from .prepare import view_counts
from sbfnav.dataset import CityReferCatalog
from collections import Counter


def main():
    source=ROOT/'runs/a2_waldo_landmark_20261027/cache_smoke'
    output=ROOT/'runs/a2_waldo_landmark_20261027/cache_smoke_car_anchor_position_v2'
    output.mkdir(exist_ok=True)
    files=sorted(source.glob('train_*.pt'))
    signature=dict(scope='TRAIN smoke labels only, no formal cache/training',
                   public_parent_signature_sha256=sha(source/'signature.json'),
                   parent_rows={p.name:sha(p) for p in files},
                   private_label_implementation_sha256=sha(ROOT/'experiments/a2_waldo_landmark_belief/private_labels.py'))
    save(signature,output/'signature.json')
    digest=sha(output/'signature.json')
    index=PrivateAssociationIndex(CityReferCatalog(DATA/'cityrefer'))
    counts=Counter()
    for path in files:
        row=torch.load(path,map_location='cpu',weights_only=False)
        parent_public=row['public']
        row=copy.deepcopy(row)
        y=row['labels']
        truth,known,details=index.associate(row['target_instance'][0],row['meta']['candidates'],row['target_instance'][1],'train')
        offset=row['labels']['target_xy'][None].numpy()-row['public']['spatial']['candidate_xy'].numpy()
        limit=row['public']['network']['offset_limit'].numpy()
        valid=(truth>0) & (np.abs(offset)<=limit).all(-1) & row['private']['target_visible']
        match=bool(valid.any())
        match_valid=match or (len(row['meta']['candidates'])>0 and known.all() and not (truth>0).any())
        y.update(ranking=torch.from_numpy(truth),rank_label_mask=torch.from_numpy(known),
                 offset=torch.from_numpy(offset),offset_valid=torch.from_numpy(valid),
                 match=torch.tensor(float(match)),match_valid=torch.tensor(bool(match_valid)))
        row['private']['association']=details
        row['signature_sha256']=digest
        row['parent_row_sha256']=signature['parent_rows'][path.name]
        assert all(torch.equal(value,row['public'][g][k]) for g,values in parent_public.items() for k,value in values.items())
        torch.save(row,output/path.name)
        counts.update(view_counts(row))
    save(dict(scope=signature['scope'],signature_sha256=digest,counts=dict(counts),
              public_tensors_unchanged=True,formal_training_complete=False),
         ROOT/'reports/a2_execution_20261027/relabelled_train_smoke.json')
    print(dict(counts),flush=True)


if __name__=='__main__':
    main()
