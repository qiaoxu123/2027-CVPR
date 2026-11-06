"""Read-only supervisor: archive actual completed-epoch weights, never restart fitting."""
import argparse,json,subprocess,time,shutil
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256
from scripts.unified_pgvb_v3_pilot import V3_FORMAT

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--session',required=True);a=p.parse_args();folder=a.run/'train';archive=folder/'epochs';archive.mkdir(exist_ok=True,parents=True)
    index_path=archive/'index.json';records=json.loads(index_path.read_text()) if index_path.exists() else []
    seen={x['epoch'] for x in records};last_stamp=None
    while True:
        active=subprocess.run(['tmux','has-session','-t','='+a.session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        path=folder/'resume.pt'
        if path.exists():
            stamp=path.stat().st_mtime_ns
            if stamp!=last_stamp:
                last_stamp=stamp;state=torch.load(path,map_location='cpu',weights_only=True);ep=state['epoch']
                if state.get('next_offset',0)==0 and ep>0 and ep not in seen and state['losses'] and state['losses'][-1]['epoch']==ep:
                    manifest=dict(source_v2_sha256=state['source_sha256'],cache_manifest_sha256=state['cache_sha256'],initialize_v3_sha256=state.get('initialize_v3_sha256'),seed=state['seed'],fit=len(state['fit']),dev=len(state['dev']),epochs=ep,losses=state['losses'],origin='actual completed epoch resume state; copied without model execution',evaluation_selection='current full evaluation remains final last.pt, not this snapshot',visual_encoder_frozen=True,val_test_access=False)
                    target=archive/f'epoch_{ep:03d}.pt';atomic_save(dict(format=V3_FORMAT,model=state['model'],manifest=manifest),target)
                    entry=dict(epoch=ep,train_loss=state['losses'][-1]['mean_train_loss'],checkpoint=str(target),sha256=file_sha256(target));records.append(entry);seen.add(ep)
                    best=min(records,key=lambda x:x['train_loss']);temporary=folder/'best_train_loss.pt.tmp';shutil.copyfile(best['checkpoint'],temporary);temporary.replace(folder/'best_train_loss.pt')
                    atomic_json(records,index_path);atomic_json(best,folder/'best_train_loss.json');print(json.dumps(entry),flush=True)
        if not active:break
        time.sleep(3)
if __name__=='__main__':main()
