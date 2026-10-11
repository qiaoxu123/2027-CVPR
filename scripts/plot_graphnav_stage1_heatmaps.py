"""Replay saved public observations and plot actual dense goal belief fields."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/graphnav_v1_stage1')
    parser.add_argument('--output', default='runs/graphnav_v1_stage1/heatmaps')
    args = parser.parse_args()
    root, out = Path(args.run), Path(args.output)
    if out.exists():
        raise ValueError('Refusing to overwrite diagnostic output')
    out.mkdir(parents=True)
    reports = {split: json.loads((root/f'full_{split}/metrics.json').read_text())
               for split in ('seen', 'unseen')}
    settings = reports['seen']['args']
    sys.path.insert(0, settings['sbf_source'])
    from sbfnav.config import load_config
    from sbfnav.dataset import CityNavDataset, CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from graphnav.closedloop import PublicEpisode, PublicCatalog
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.stage1_runtime import GraphNavStage1Policy

    torch.manual_seed(0)
    np.random.seed(0)
    data = Path(settings['data_root'])
    config = load_config(settings['config'])
    catalog = CityReferCatalog(data/'cityrefer')
    builder = PlanningStateBuilder(data, PublicCatalog(catalog),
        canvas_size=config['model']['canvas_size'], field_size=config['model']['field_size'],
        max_height_above_ground_m=config['state']['max_height_above_ground_m'],
        trajectory_radius_m=config['state']['trajectory_radius_m'])
    builder = NativeObservationBuilder(builder, data/'rgbd', out/'observations')
    policy = GraphNavStage1Policy(root/'best.pt', out/'public_inputs')
    snapshots = []

    def capture(module, inputs, result):
        snapshots.append({name: tensor.detach().float().cpu().numpy()[0].copy()
            for name, tensor in [('static', result.static.belief_logits),
                                 ('belief', result.belief.belief_logits), ('delta', result.delta)]})

    handle = policy.model.register_forward_hook(capture)
    manifest = []
    for split in ('seen', 'unseen'):
        dataset = CityNavDataset(data, 'val_'+split, version=config['data']['version'])
        rows = [json.loads(line) for line in (root/f'full_{split}/predictions.jsonl').read_text().splitlines()]
        # Deterministic examples, including failures; choose different maps where possible.
        failed = next(row for row in rows if row['metrics']['SR'] == 0)
        successful = next((row for row in rows if row['metrics']['SR'] == 1
                           and row['map_name'] != failed['map_name']),
                          next(row for row in rows if row['metrics']['SR'] == 1))
        for label, row in [('success', successful), ('failure', failed)]:
            source = dataset[row['index']]
            public = PublicEpisode(source.map_name, source.instruction, source.start_pose.copy(),
                tuple(catalog.referenced_landmarks(source)), tuple(catalog.referenced_landmark_names(source)))
            policy.set_episode(public, builder.transform(public.map_name), catalog)
            snapshots.clear()
            poses = np.asarray(row['poses'])
            for diagnostic in row['step_diagnostics']:
                step = diagnostic['observation_step']
                state = builder.build_from_poses(public, poses[:step+1])
                prediction = policy.predict(state, public.instruction)
                for key in ('static_top16_world_xy', 'belief_top16_world_xy', 'selected_goal_xy'):
                    np.testing.assert_allclose(prediction.diagnostics[key], diagnostic[key], atol=1e-4, rtol=0)
                np.testing.assert_allclose(prediction.diagnostics['stop_probability'], diagnostic['stop_probability'], atol=1e-6, rtol=0)
            # Labels only enter plotting after every public-state forward has completed.
            target = np.asarray(source.target_position)[:2]
            first, last = snapshots[0], snapshots[-1]
            fields = [first['static'], first['belief'], last['belief']]
            probabilities = [torch.from_numpy(field).softmax(-1).numpy() for field in fields]
            grid = policy.world[0].float().cpu().numpy().reshape(policy.args.grid_size, policy.args.grid_size, 2)
            shape = grid.shape[:2]
            vmax = max(float(p.max()) for p in probabilities)
            vmin = max(min(float(p.min()) for p in probabilities), vmax*1e-5, 1e-12)
            fig, axes = plt.subplots(1, 4, figsize=(19, 5.5), layout='constrained')
            titles = ['Initial static belief', 'First observed belief (step 0)',
                      f"Final observed belief (step {row['step_diagnostics'][-1]['observation_step']})",
                      'Final visual/history residual (logit)']
            for i, ax in enumerate(axes):
                if i < 3:
                    artist = ax.pcolormesh(grid[..., 0], grid[..., 1], probabilities[i].reshape(shape),
                        shading='nearest', cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax))
                    top = policy.world[0, int(fields[i].argmax())].cpu().numpy()
                    ax.scatter(*top, marker='x', c='#00ffff', s=95, linewidths=2, label='Heatmap maximum')
                else:
                    artist = ax.pcolormesh(grid[..., 0], grid[..., 1], last['delta'].reshape(shape),
                        shading='nearest', cmap='RdBu_r', vmin=-.5, vmax=.5)
                    fig.colorbar(artist, ax=ax, shrink=.75, label='Bounded logit correction')
                path = poses if i >= 2 else poses[:1]
                ax.plot(path[:, 0], path[:, 1], '-o', c='#35c9ff', ms=2, lw=1.5, label='Executed path')
                ax.scatter(*poses[0, :2], marker='^', c='lime', edgecolor='black', s=65, label='Start')
                ax.scatter(*target, marker='*', c='red', edgecolor='white', s=140, label='GT (plot only)')
                ax.add_patch(plt.Circle(target, 20, fill=False, color='red', lw=1.2))
                ax.set_title(titles[i], fontsize=10)
                ax.set_aspect('equal')
                ax.set_xlabel('World East X (m)')
                ax.set_ylabel('World North Y (m)')
            fig.colorbar(axes[0].collections[0], ax=list(axes[:3]), shrink=.75, label='Goal probability (shared log scale)')
            axes[0].legend(loc='lower left', fontsize=6)
            fig.suptitle(f"{split.upper()} / {label} | {source.map_name} | episode {row['index']} | "
                f"SR={int(row['metrics']['SR'])}, NE={row['metrics']['NE']:.2f}m\n"
                + '\n'.join(textwrap.wrap(source.instruction, width=145)), fontsize=10)
            name = f"{split}_{label}_{row['index']}"
            fig.savefig(out/(name+'.png'), dpi=160)
            plt.close(fig)
            np.savez_compressed(out/(name+'.npz'), grid_world_xy=grid, static_logits=first['static'],
                first_belief_logits=first['belief'], final_belief_logits=last['belief'],
                final_delta=last['delta'], target_xy=target, poses=poses)
            manifest.append(dict(split=split, group=label, index=row['index'], map_name=source.map_name,
                image=name+'.png', arrays=name+'.npz', metrics=row['metrics'], replay_matches_saved_diagnostics=True))
            print(json.dumps(manifest[-1]), flush=True)
    handle.remove()
    metrics = {split: report['metrics'] for split, report in reports.items()}
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    with (out/'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['split', *metrics['seen'].keys()])
        writer.writeheader()
        writer.writerows(dict(split=split, **values) for split, values in metrics.items())
    (out/'manifest.json').write_text(json.dumps(dict(cases=manifest,
        checkpoint_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),
        selection='First failure and first success on a different map, in source episode order.',
        note='Actual 28x28 network fields replayed from saved causal observations. GT used only after replay for plotting. Shared probability scale within each figure; residual is a logit correction, not probability.'), indent=2))


if __name__ == '__main__':
    main()
