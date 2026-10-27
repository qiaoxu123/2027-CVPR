"""Full-cache gates, explicit supervision metrics and fixed 20+5 protocol."""
import json
from collections import Counter
from pathlib import Path
from .paths import sha

RANK_EPOCHS=20
REFINE_EPOCHS=5
TRAIN_CONFIG=dict(seed=0,batch_size=8,workers=2,lr=.0003,weight_decay=.01,
                  rank_epochs=RANK_EPOCHS,refine_epochs=REFINE_EPOCHS,
                  gradient_clip=5.,selection='ranking: MRR then dev loss; refinement: MRR then offset MAE then dev loss',
                  rank_loss_weight=1.,direction_weight=.2,distance_weight=.2,trust_weight=.2,offset_weight=.25)


def validate_complete_cache(directory,source,*,hash_rows=True):
    """Require every source episode and verify all row hashes, not just a flag."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('complete') is not True:
        raise ValueError('Full training requires a complete cache; smoke is forbidden')
    signature=sha(directory/'signature.json')
    episode_rows=[]
    for entry in source['entries']:
        completed=directory/f"{entry['split']}_{entry['index']}_complete.json"
        if not completed.exists():
            raise ValueError(f'Missing source episode completion: {completed.name}')
        info=json.loads(completed.read_text())
        if info['signature_sha256']!=signature:
            raise ValueError('Episode completion signature mismatch')
        for item in info['entries']:
            episode_rows.append(dict(item,split=entry['split'],episode=entry['index']))
    def identities(entries):
        return Counter((e['split'],e['episode'],e['filename'],e['step'],e['sha256']) for e in entries)
    if identities(episode_rows)!=identities(manifest['entries']):
        raise ValueError('Full manifest does not match all source episode rows')
    filenames=[e['filename'] for e in manifest['entries']]
    if len(filenames)!=len(set(filenames)):
        raise ValueError('Duplicated cache view filenames')
    if manifest['signature_sha256']!=signature:
        raise ValueError('Full cache signature mismatch')
    counts=Counter(e['split'] for e in manifest['entries'])
    if not all(counts[p]>0 for p in ('train','dev','calibration')):
        raise ValueError('Full train/dev/calibration view populations required')
    for number,entry in enumerate(manifest['entries'],1):
        path=directory/entry['filename']
        if not path.exists() or (hash_rows and sha(path)!=entry['sha256']):
            raise ValueError(f'Cache row missing or changed: {path}')
        if number%2000==0:
            print(f'Verified cache {number}/{len(filenames)}',flush=True)
    return manifest


def ranking_metrics(scores,truth,known,reliable):
    """MRR/top1 among VERIFIED candidates on genuine positive/negative views.

    This is a training diagnostic, not full-pool localization Recall or SR.
    Unknown boxes are excluded from the supervised ranking population.
    """
    reciprocal=[];correct=[]
    for s,y,k,r in zip(scores,truth,known,reliable):
        valid=k.bool() & r.bool()
        if not (valid & (y>0)).any() or not (valid & (y<=0)).any():
            continue
        order=s[valid].argsort(descending=True,stable=True)
        positive=(y[valid][order]>0).nonzero().flatten()
        rank=int(positive[0])+1
        reciprocal.append(1./rank);correct.append(float(rank==1))
    return reciprocal,correct


def selection_key(metrics,*,refine=True):
    mrr=metrics['known_candidate_mrr']
    offset=metrics['offset_mae_m']
    return (mrr if mrr is not None else -1.,(-offset if offset is not None else -1e9) if refine else 0.,-metrics['loss'])
