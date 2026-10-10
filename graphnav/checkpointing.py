"""Atomic experiment checkpoints with input fingerprints and complete RNG state."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import numpy as np
import torch


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def input_fingerprint(args, keys):
    return {key:dict(path=str(Path(getattr(args,key)).resolve()),
                    sha256=file_sha256(getattr(args,key))) for key in keys}


def rng_state():
    state=np.random.get_state()
    return dict(python=random.getstate(),torch=torch.get_rng_state(),
        numpy=(state[0],state[1].tolist(),state[2],state[3],state[4]),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);torch.set_rng_state(state['torch'].cpu())
    s=state['numpy'];np.random.set_state((s[0],np.array(s[1],dtype=np.uint32),s[2],s[3],s[4]))
    if state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def provenance():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    diff=subprocess.check_output(['git','diff','HEAD','--binary'])
    return dict(commit=commit,working_diff_sha256=hashlib.sha256(diff).hexdigest())


def atomic_save(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name+'.',suffix='.tmp',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
    try:
        torch.save(payload,temporary);os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2))
    os.replace(temporary,path)


def restore_training(path,model,optimizer,args,fingerprint,*,ignored=('resume','epochs','device','workers')):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('input_fingerprint') != fingerprint:
        raise ValueError('checkpoint source SHA256/path mismatch; refuse resume')
    prior=state['args'];current=vars(args)
    for key in sorted(set(prior)|set(current)):
        if key not in ignored and prior.get(key)!=current.get(key):
            raise ValueError('resume configuration differs: '+key)
    if state['epoch']>args.epochs:raise ValueError('checkpoint is beyond requested epochs')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return state
