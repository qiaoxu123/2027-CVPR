from argparse import Namespace
import random
import numpy as np
import pytest
import torch
from graphnav.checkpointing import (atomic_save,rng_state,restore_training,input_fingerprint)


def test_optimizer_and_all_rng_resume_reproduce_training_and_reject_changed_data(tmp_path):
    torch.manual_seed(7);random.seed(7);np.random.seed(7)
    model=torch.nn.Sequential(torch.nn.Linear(2,8),torch.nn.Dropout(.2),torch.nn.Linear(8,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    path=tmp_path/'data';path.write_text('version1')
    args=Namespace(train=str(path),epochs=2,lr=.01,seed=7,resume=None)
    fingerprint=input_fingerprint(args,('train',))
    def update():
        x=torch.randn(8,2)*(random.random()+np.random.rand())
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
    update()
    checkpoint=tmp_path/'last.pt'
    atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=1,
        args=vars(args),input_fingerprint=fingerprint,rng=rng_state()),checkpoint)
    update();expected={k:v.clone() for k,v in model.state_dict().items()}
    restore_training(checkpoint,model,optimizer,args,fingerprint)
    update()
    for key,value in model.state_dict().items():assert torch.equal(value,expected[key])
    path.write_text('version2')
    with pytest.raises(ValueError,match='SHA256'):
        restore_training(checkpoint,model,optimizer,args,input_fingerprint(args,('train',)))
    args.lr=.1
    with pytest.raises(ValueError,match='configuration'):
        restore_training(checkpoint,model,optimizer,args,fingerprint)
def test_concurrent_atomic_writers_publish_valid_payloads(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import torch
    from graphnav.checkpointing import atomic_save
    original=torch.save;barrier=threading.Barrier(2);temporary=[]
    def synchronized_save(payload,path):
        temporary.append(str(path));original(payload,path);barrier.wait(timeout=5)
    monkeypatch.setattr(torch,'save',synchronized_save)
    path=tmp_path/'shared.pt'
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i:atomic_save({'value':torch.tensor(i)},path),range(2)))
    assert len(set(temporary))==2
    assert torch.load(path,weights_only=True)['value'].item() in (0,1)
    assert not list(tmp_path.glob('*.tmp'))

