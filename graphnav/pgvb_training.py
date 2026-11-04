"""PGVB training inputs and weak losses, with explicit same-image hard negatives."""
import collections
import torch
from torch.nn import functional as F


def pgvb_batch(data,indices,device='cuda',control='real'):
    rows=[data['records'][i] for i in indices];frames=[data['frames'][r['frame_id']] for r in rows]
    queries=[data['queries'][r['query_key']] for r in rows]
    images=torch.stack([f['image_features'] for f in frames]).to(device)
    valid=torch.stack([f['valid'] for f in frames]).to(device)
    if control=='zero_rgb':images.zero_()
    if control=='shuffle_rgb':
        for i in range(len(images)):
            ids=valid[i].nonzero().flatten();images[i,ids]=images[i,ids.flip(0)].clone()
    poly=torch.stack([r['polygon_tokens'] for r in rows]).to(device)
    name=torch.stack([q['name_text'] for q in queries]).to(device);typ=torch.stack([q['type_text'] for q in queries]).to(device)
    if control in ('wrong_polygon','wrong_name'):
        for i,r in enumerate(rows):
            if not r['hard_negative_indices']:raise ValueError('No controlled same-image replacement')
            other=data['records'][r['hard_negative_indices'][0]]
            if control=='wrong_polygon':poly[i]=other['polygon_tokens'].to(device)
            else:name[i]=data['queries'][other['query_key']]['name_text'].to(device)
    if control=='polygon_zero':poly.zero_()
    return dict(image_features=images,name_text=name,type_text=typ,polygon_tokens=poly,valid=valid),torch.stack([r['labels'] for r in rows]).to(device),rows


def pgvb_loss(scores,labels,rows):
    positive=labels==1;negative=labels==0;known=positive|negative
    if not positive.any(-1).all() or not negative.any(-1).all():raise ValueError('Ambiguous/empty supervision must not be forced')
    region=(torch.logsumexp(scores.masked_fill(~known,float('-inf')),-1)-torch.logsumexp(scores.masked_fill(~positive,float('-inf')),-1)).mean()
    bce=((F.softplus(-scores)*positive).sum(-1)/positive.sum(-1)+(F.softplus(scores)*negative).sum(-1)/negative.sum(-1)).mean()
    pairs=[]
    for i,a in enumerate(rows):
        own=scores[i][positive[i]].mean()
        for j,b in enumerate(rows):
            if i==j or a['frame_id']!=b['frame_id'] or a['type']!=b['type'] or (positive[i]&positive[j]).any():continue
            pairs.append(F.softplus(1+scores[i][positive[j]].mean()-own))
    pair=torch.stack(pairs).mean() if pairs else scores.sum()*0
    return dict(total=region+.2*bce+.3*pair,region=region,bce=bce,pair=pair,pairs=len(pairs))


def frame_batches(data,subset,batch_frames,rng=None):
    grouped=collections.defaultdict(list)
    for i,r in enumerate(data['records']):
        if r['subset']==subset:grouped[r['frame_id']].append(i)
    frames=sorted(grouped)
    if rng:rng.shuffle(frames)
    for start in range(0,len(frames),batch_frames):yield [i for f in frames[start:start+batch_frames] for i in grouped[f]]
