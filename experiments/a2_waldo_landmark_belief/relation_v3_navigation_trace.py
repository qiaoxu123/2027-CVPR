"""Committed public RGB/native-grid artifacts and inference-only latency."""
import hashlib,json,shutil,base64,zlib
from pathlib import Path
import cv2
import numpy as np
import torch
from .paths import sha
from .train import atomic_checkpoint


def save_belief(captured,directory,*,index,step,signature):
    directory=Path(directory);folder=directory/'belief';folder.mkdir(exist_ok=True)
    path=folder/f'episode_{index:06d}_step_{step:02d}.pt'
    if path.exists():
        archive=directory/'partial_replay_archive';archive.mkdir(exist_ok=True)
        shutil.copyfile(path,archive/(path.stem+'_'+sha(path)+'.pt'))
    public={k:v.detach().cpu().clone() for k,v in captured.items()}
    atomic_checkpoint(dict(index=index,step=step,signature_sha256=signature,**public),path)
    return dict(filename=str(path.relative_to(directory)),sha256=sha(path))


def local_artifact(directory,name,folder):
    path=Path(name)
    if len(path.parts)!=2 or path.parts[0]!=folder or path.name in ('.','..'):
        raise ValueError('Unsafe public observation artifact path')
    root=Path(directory).resolve();resolved=(root/path).resolve()
    if resolved.parent!=root/folder:raise ValueError('Public observation artifact escapes its directory')
    return resolved


def validate_observed_artifacts(row,directory,*,bounds,ground):
    from graphnav.geometry import grid_centers
    from graphnav.memory import ObservedRGBMemory
    diagnostics=row['step_diagnostics'];observations=row['observations']
    if (not diagnostics or len(observations)!=len(diagnostics) or len(row['inference_seconds'])!=len(diagnostics)
        or len(row['decision_seconds'])!=len(diagnostics)):
        raise ValueError('Incomplete observed inference records')
    times=np.asarray(row['inference_seconds'],dtype=float)
    if not np.isfinite(times).all() or (times<=0).any():raise ValueError('Invalid synchronized inference timing')
    poses=np.asarray(row['poses'],dtype=float);bounds=torch.as_tensor(bounds,dtype=torch.float32)
    if poses.ndim!=2 or poses.shape[1]!=4 or not np.isfinite(poses).all() or len(poses)<len(diagnostics):raise ValueError('Incomplete public decision poses')
    if bounds.shape!=(4,) or not torch.isfinite(bounds).all() or (bounds[2:]<=bounds[:2]).any():raise ValueError('Original public map bounds required')
    expected_world=(bounds[:2]+grid_centers(28)*(bounds[2:]-bounds[:2]))[None]
    for step,(d,o) in enumerate(zip(diagnostics,observations)):
        if d['observation_step']!=step:raise ValueError('Observed decision steps are incomplete or reordered')
        image=o['native_rgb'];path=local_artifact(directory,image['filename'],'rgb')
        if path.name!=image['bgr_sha256']+'.png' or sha(path)!=image['sha256']:raise ValueError('Observed RGB bytes changed')
        bgr=cv2.imread(str(path))
        if bgr is None or bgr.shape!=(384,384,3) or hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()!=image['bgr_sha256']:
            raise ValueError('Observed RGB pixels changed')
        packed=zlib.decompress(base64.b64decode(o['valid_bitmap']))
        expected_bits=np.packbits((bgr.max(-1)>0).reshape(-1)).tobytes()
        if o['shape']!=[384,384] or packed!=expected_bits:raise ValueError('Sensor bitmap differs from actual native RGB')
        pose=poses[step];footprint=ObservedRGBMemory.view_corners(pose[:2],pose[3],pose[2]-ground)
        if not np.array_equal(np.asarray(o['footprint_world']),footprint):raise ValueError('Observed footprint differs from actual public decision pose')
        artifact=o['native_belief'];path=local_artifact(directory,artifact['filename'],'belief')
        if path.name!=f"episode_{row['index']:06d}_step_{step:02d}.pt" or sha(path)!=artifact['sha256']:
            raise ValueError('Native belief source/bytes changed')
        public=torch.load(path,map_location='cpu',weights_only=True)
        if (public['index']!=row['index'] or public['step']!=step or public['signature_sha256']!=row['signature_sha256']):
            raise ValueError('Native belief episode/step/signature changed')
        world=public['world']
        if world.shape!=(1,784,2) or not torch.isfinite(world).all():raise ValueError('Native28 world grid required')
        if not torch.allclose(world.float(),expected_world,atol=1e-3,rtol=0):raise ValueError('Native28 grid differs from original public map bounds')
        for name,diag in (('static','static_top16_world_xy'),('belief','belief_top16_world_xy')):
            logits=public[name+'_logits'];ids=public[name+'_ids']
            if (logits.shape!=(1,784) or not torch.isfinite(logits).all() or ids.shape!=(1,16)
                or ids.dtype!=torch.long or ids.unique().numel()!=16 or ids.min()<0 or ids.max()>=784):
                raise ValueError('Invalid native belief/ranking shape')
            chosen=logits.float().gather(1,ids)[0]
            if (chosen[:-1]<chosen[1:]).any() or chosen.min()<logits.float().topk(16,-1).values.min():
                raise ValueError('Saved actual ranking is not a native top16 selection')
            if not np.array_equal(world[0,ids[0]].numpy(),np.asarray(d[diag])):
                raise ValueError('Recorded retrieval differs from saved native belief')
    return True


def latency_summary(rows):
    values=np.asarray([t for row in rows for t in row['inference_seconds']],dtype=float)
    if not values.size or not np.isfinite(values).all() or (values<=0).any():raise ValueError('Complete finite synchronized inference timings required')
    values=values*1000
    return dict(decisions=len(values),mean_ms=float(values.mean()),median_ms=float(np.median(values)),
        p95_ms=float(np.quantile(values,.95)),
        definition='Per-decision synchronized policy.predict, including native evidence capture hook; excludes PNG/PT/trace serialization and metric computation; first-query encoding included',
        cache_caveat='Fresh and resumed episodes may have different cache warmth; do not claim a controlled cold-start hardware benchmark')
