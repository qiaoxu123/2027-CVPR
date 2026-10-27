"""Pad variable observed box/reference/token axes without manufacturing validity."""
import torch


def _stack(values):
    first=values[0]
    if first.ndim==0:
        return torch.stack(values)
    shape=tuple(max(v.shape[j] for v in values) for j in range(first.ndim))
    result=torch.zeros((len(values),*shape),dtype=first.dtype)
    for i,value in enumerate(values):
        result[(i,*[slice(0,n) for n in value.shape])]=value
    return result


def collate_views(rows):
    def group(parts):
        return {key:_stack([p[key] for p in parts]) for key in parts[0]}
    return dict(network=group([r['public']['network'] for r in rows]),
                spatial=group([r['public']['spatial'] for r in rows]),
                labels=group([r['labels'] for r in rows]))


def to_device(batch,device):
    return {group:{key:value.to(device) for key,value in values.items()} for group,values in batch.items()}
