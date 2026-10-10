"""Seen-only static predictions for frozen controlled models; labels read after forward."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.full_experiment import new_model, _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.static_controls import graph_control
from graphnav.static_localization import cross_entropy_goal
from graphnav.checkpointing import file_sha256, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', required=True)
    parser.add_argument('--bert-cache', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=False, parents=True)
    encoder = FrozenBERTTokens('bert-base-uncased', device='cuda', max_length=96)
    encoder.load_cache(args.bert_cache)
    report = dict(models={}, provenance=provenance(), input_sha256=file_sha256(args.records))
    for entry in args.checkpoints:
        name, path = entry.split('=', 1)
        state = torch.load(path, map_location='cpu', weights_only=True)
        cfg = SimpleNamespace(**state['args'])
        model = new_model(cfg, torch.device('cuda')).eval()
        model.load_state_dict(state['model'], strict=True)
        dataset = FullCandidateDataset(args.records, grid_size=cfg.grid_size,
            max_landmarks=cfg.max_landmarks, max_roads=cfg.max_roads,
            max_relations=cfg.max_relations, prior_sigma_m=cfg.prior_sigma_m,
            rule_sigma_m=cfg.rule_sigma_m)
        loader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_full)
        losses, distances = [], []
        offset = 0
        with torch.inference_mode(), (output / (name + '_predictions.jsonl')).open('w') as stream:
            for batch in loader:
                inputs, tokens, mask, names = _batch_inputs(batch, encoder, torch.device('cuda'), 'geometry')
                inputs = graph_control(inputs, cfg.graph_control if hasattr(cfg, 'graph_control') else 'full', batch['node_names'])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    field = model(inputs, variant=state.get('variant', 'geometry'), tokens=tokens, token_mask=mask, node_emb=names)
                # Private labels are accessed only after the public forward.
                target = batch['target_xy'].to('cuda')
                loss = cross_entropy_goal(field.belief_logits, target, grid_size=cfg.grid_size)
                top = field.belief_logits.topk(8, dim=-1).indices
                xy = inputs['grid_xy'].gather(1, top[..., None].expand(-1, -1, 2))
                distance = ((xy - target[:, None]) * batch['map_sizes'].to('cuda')[:, None]).norm(dim=-1).cpu()
                distances.append(distance)
                losses.append((float(loss), len(target)))
                for index, row in enumerate(distance.tolist()):
                    record = dataset.records[offset + index]
                    stream.write(json.dumps(dict(index=offset+index, map_name=record['map_name'],
                        instruction=record['instruction'], top8_cells=top[index].cpu().tolist(),
                        distance_m=row, NE_m=row[0], **{f'R@{k}':min(row[:k]) <= 20 for k in (1,4,8)})) + '\n')
                offset += len(target)
        distance = torch.cat(distances)
        report['models'][name] = dict(checkpoint=path, checkpoint_sha256=file_sha256(path), epoch=state['epoch'],
            samples=offset, loss=sum(loss*n for loss,n in losses)/offset,
            NE_m=float(distance[:,0].mean()), **{f'R@{k}/20m':float((distance[:,:k].min(-1).values <= 20).float().mean()) for k in (1,4,8)})
        (output / 'metrics.json').write_text(json.dumps(report, indent=2))
        print(name + ' complete', flush=True)
        del model


if __name__ == '__main__':
    main()
