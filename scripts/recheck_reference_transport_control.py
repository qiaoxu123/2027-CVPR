"""Correct only a frozen wrong-frame ablation; do not retrain or overwrite."""
import argparse
import json
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.reference_evidence_transport import ReferenceTransportPositionHead
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import records
from scripts.train_reference_evidence_transport import evaluate,wrong_frame_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','suite','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior correction')
    status=json.loads((a.suite/'status.json').read_text())
    if status['status']!='complete':raise ValueError('only terminal frozen suite')
    torch.set_num_threads(4);r=a.root;roles=torch.load(r/'reference_vectors_v2.pt',map_location='cpu',weights_only=True)
    dev=records(r/'chain_train400_features',r/'fixed_yaw_train400_s1/labels_dev.jsonl','dev',roles,'reference');donors,partners=wrong_frame_features(dev)
    output={}
    for mode in ('local','global','constant'):
        path=a.suite/(mode+'_s0')/'final.pt';before=file_sha256(path);state=torch.load(path,map_location='cpu',weights_only=True)
        model=ReferenceTransportPositionHead(WorldLikelihoodPositionHead(),mode=mode).cuda().eval();model.load_state_dict(state['model'],strict=True)
        result=evaluate(model,dev,'wrong-frame',donors)
        assert before==file_sha256(path)
        output[mode]=dict(checkpoint_sha256=before,result=result)
    a.output.mkdir(parents=True)
    atomic_json(dict(results=output,partners=partners,eval_source_sha256=file_sha256(Path('scripts/train_reference_evidence_transport.py')),
        warning='Original wrong-frame columns were invalid due to omitted forwarding. This frozen-only rerun replaces that diagnostic column; no model selection or retraining.'),a.output/'wrong_frame_corrected.json')
    print(json.dumps({m:x['result']['metrics'] for m,x in output.items()}))


if __name__=='__main__':main()
