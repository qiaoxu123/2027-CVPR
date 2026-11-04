"""Read-only paired analysis of frozen named-landmark rollouts (no inference)."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

METHODS = ('Geometry', 'LegacyPGVB', 'TrackedPGVB')


def summarize_episode(row):
    m = row['metrics']
    ds = row['step_diagnostics']
    goals = np.array([d['selected_goal_xy'] for d in ds])
    jumps = np.linalg.norm(np.diff(goals, axis=0), axis=1)
    stop = row['termination'] == 'policy_stop'
    last_good = ds[-1]['public_landmark_goal_distance_m'] <= 20
    if m['SR']:
        cause = 'success'
    elif m['OSR']:
        cause = 'entered_then_left_wrong_goal' if not last_good else 'entered_then_left_stop_or_control'
    elif not any(d['public_landmark_goal_distance_m'] <= 20 for d in ds):
        cause = 'never_selected_correct_goal'
    elif last_good:
        cause = 'correct_goal_but_control_or_horizon'
    else:
        cause = 'correct_goal_lost_before_arrival'
    states = [d.get('landmark_tracking_state', 'geometry') for d in ds]
    return dict(
        SPL_polygon=m['SR'] * m['initial_polygon_distance_m'] / max(m['path_length_m'], m['initial_polygon_distance_m'], 1e-9),
        correct_stop=int(stop and m['SR']), false_stop=int(stop and not m['SR']),
        first_entry_step=m['first_entry_step'], entered_then_left=int(m['OSR'] and not m['SR']),
        goal_jumps_over20=int((jumps > 20).sum()), max_goal_jump_m=float(jumps.max()) if len(jumps) else 0.,
        tracking_frames=states.count('tracking'), tracking_episode=int('tracking' in states),
        fallback_events=sum(a in ('tracking', 'coasting') and b == 'geometry' for a, b in zip(states, states[1:])),
        coasting_frames=states.count('coasting'),
        visual_active_frames=sum(d.get('pgvb_active', False) for d in ds), decisions=len(ds),
        rgb_encoded_frames=sum(d.get('siglip2_rgb_forward', False) for d in ds),
        mean_decision_ms=1000 * float(np.mean(row['decision_seconds'])), failure_observation=cause,
        near_geometry_frames=sum(d.get('near_landmark_geometry', {}).get('within_20m', False) for d in ds),
    )


def paired_changes(rows, reference, treatment):
    groups = {m: {r['index']: r for r in rows if r['method'] == m} for m in (reference, treatment)}
    assert groups[reference].keys() == groups[treatment].keys()
    corrected, damaged = [], []
    difference = []
    for i, r in sorted(groups[reference].items()):
        a, b = r['metrics']['SR'], groups[treatment][i]['metrics']['SR']
        difference.append(b-a)
        if b > a: corrected.append(i)
        if b < a: damaged.append(i)
    rng = np.random.default_rng(42)
    samples = rng.choice(difference, size=(10000, len(difference)), replace=True).mean(1)
    return dict(reference=reference, treatment=treatment, corrected=corrected, damaged=damaged,
                net=len(corrected)-len(damaged), SR_delta=float(np.mean(difference)),
                episode_paired_bootstrap_CI95=np.quantile(samples, [.025, .975]).tolist(),
                caveat='固定64条具名代理任务的Episode配对区间，不是场景泛化或原CityNav SR结论')


def draw_polygon(ax, region, **kwargs):
    for p in getattr(region, 'geoms', [region]):
        x, y = p.exterior.xy
        ax.plot(x, y, **kwargs)


def main():
    global METHODS
    # Plotting is optional for read-only metric tests and headless CPU CI.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs=3, default=list(METHODS))
    a = p.parse_args()
    METHODS = tuple(a.methods)
    if len(set(METHODS)) != 3:
        raise ValueError('Three distinct paired methods required')
    rows = [json.loads(x) for x in (a.run/'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == 192 and len({(r['method'], r['index']) for r in rows}) == 192
    tasks = {t['index']: t for t in json.loads((a.selection/'tasks_public.json').read_text())}
    for method in METHODS:
        assert {r['index'] for r in rows if r['method'] == method} == set(tasks)
    a.output.mkdir(parents=True, exist_ok=True)
    details = [{**r, 'analysis': summarize_episode(r)} for r in rows]
    summary = {}
    for method in METHODS:
        group = [r for r in details if r['method'] == method]
        ds = [d for r in group for d in r['step_diagnostics']]
        summary[method] = dict(
            episodes=len(group), successes=sum(r['metrics']['SR'] for r in group),
            **{k: float(np.mean([r['metrics'][k] for r in group])) for k in ('SR', 'OSR', 'NE', 'steps', 'path_length_m')},
            **{k: float(np.mean([r['analysis'][k] for r in group])) for k in ('SPL_polygon', 'mean_decision_ms')},
            **{k: sum(r['analysis'][k] for r in group) for k in ('correct_stop', 'false_stop', 'entered_then_left', 'goal_jumps_over20', 'tracking_frames', 'tracking_episode', 'fallback_events', 'coasting_frames', 'visual_active_frames', 'decisions', 'rgb_encoded_frames')},
            terminations=dict(Counter(r['termination'] for r in group)),
            failure_observations=dict(Counter(r['analysis']['failure_observation'] for r in group)),
            evidence_rejection_reasons=dict(Counter(d.get('landmark_acquisition', {}).get('reason', 'not_tracked_mode') for d in ds)),
            category={c: dict(n=sum(r['category'] == c for r in group), SR=float(np.mean([r['metrics']['SR'] for r in group if r['category'] == c]))) for c in sorted({r['category'] for r in group})},
        )
    paired = [paired_changes(rows, 'Geometry', m) for m in METHODS[1:]] + [paired_changes(rows, METHODS[1], METHODS[2])]
    result = dict(summary=summary, paired=paired, metrics_source='Public named Polygon proxy; no original target GT', raw_run=str(a.run), manifest=json.loads((a.run/'run_manifest.json').read_text()))
    (a.output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    with (a.output/'paired_tasks.csv').open('w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['index', 'landmark', 'category'] + [m+'_'+k for m in METHODS for k in ('SR', 'OSR', 'NE', 'first_entry_step', 'steps', 'termination')])
        for i, t in tasks.items():
            values = []
            for m in METHODS:
                r = next(r for r in rows if r['index'] == i and r['method'] == m)
                values += [r['metrics'][k] for k in ('SR', 'OSR', 'NE', 'first_entry_step', 'steps')] + [r['termination']]
            writer.writerow([i, t['landmark_name'], t['category']] + values)
    # Exact numerical trajectories and all decision diagnostics, separate from images/cache.
    (a.output/'trajectories.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in details))
    shutil.copy2(a.run/'metrics.json', a.output/'runtime_metrics.json')
    shutil.copy2(a.run/'run_manifest.json', a.output/'run_manifest.json')
    gallery = []
    for i, t in tasks.items():
        region = unary_union([Polygon(c) for c in t['public_polygons']])
        group = [next(r for r in details if r['index'] == i and r['method'] == m) for m in METHODS]
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
        for k, r in enumerate(group):
            ds = r['step_diagnostics']
            key = next((j for j, d in enumerate(ds) if d.get('landmark_tracking_state') == 'tracking'), len(ds)-1)
            record = a.run/'decisions'/f'{i:02d}_{r["method"]}'/f'step_{key:02d}'
            image = plt.imread(record.with_suffix('.png'))
            data = np.load(record.with_suffix('.npz'))
            axes[k, 0].imshow(image)
            fp = np.asarray(data['footprint_world'])
            # Projection overlay uses the actual per-frame Footprint affine basis.
            from graphnav.landmark_projection import world_to_frame
            for poly in getattr(region, 'geoms', [region]):
                pixels = world_to_frame(np.array(poly.exterior.coords), fp, image.shape)
                axes[k, 0].plot(pixels[:, 0], pixels[:, 1], color='lime', linewidth=1)
            axes[k, 0].set_title(f'{r["method"]} RGB t={key} / {ds[key].get("landmark_tracking_state")}')
            axes[k, 0].set_xlim(0, image.shape[1]); axes[k, 0].set_ylim(image.shape[0], 0)
            bounds = data['bounds']; extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
            for j, field in enumerate(('original_logits', 'final_logits'), 1):
                logits = data[field].astype(float); prob = np.exp(logits-logits.max()); prob /= prob.sum()
                axes[k, j].imshow(prob.reshape(28, 28), origin='lower', extent=extent)
                draw_polygon(axes[k, j], region, color='lime', linewidth=1)
                axes[k, j].plot(*np.vstack([fp, fp[0]]).T, '--', color='white', linewidth=.7)
                axes[k, j].set_xlim(bounds[0], bounds[2])
                axes[k, j].set_ylim(bounds[1], bounds[3])
                axes[k, j].set_title(('Geometry' if j == 1 else 'Final') + f' t={key} / terminal NE={r["metrics"]["NE"]:.2f}m')
            poses = np.asarray(r['poses']); goals = np.array([d['selected_goal_xy'] for d in ds])
            ax = axes[k, 3]
            draw_polygon(ax, region, color='green')
            ax.plot(poses[:, 0], poses[:, 1], 'o-', ms=2, label='UAV')
            ax.plot(goals[:, 0], goals[:, 1], 'x--', ms=3, label='Goal')
            ax.scatter(*poses[0, :2], color='green'); ax.scatter(*poses[-1, :2], color='red')
            ax.set_aspect('equal', adjustable='datalim'); ax.legend(fontsize=7)
            ax.set_title(f'SR={r["metrics"]["SR"]:.0f}, OSR={r["metrics"]["OSR"]:.0f}, {r["termination"]}')
        fig.suptitle(f'{i}: {t["landmark_name"]} / {t["instruction"]}', fontsize=10)
        filename = f'task_{i:02d}.png'; fig.savefig(a.output/filename, dpi=110); plt.close(fig)
        gallery.append(f'<details><summary>{i}: {html.escape(t["landmark_name"])} — '+', '.join(f'{r["method"]} SR={r["metrics"]["SR"]:.0f}' for r in group)+f'</summary><a href="{filename}"><img loading="lazy" style="width:100%" src="{filename}"></a><pre>'+html.escape(json.dumps({r['method']: r['analysis'] for r in group}, ensure_ascii=False, indent=2))+'</pre></details>')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Landmark acquisition frozen closed-loop</title><h1>64 paired public-landmark tasks</h1><p>PNG为真实逐步RGB、公开Polygon、模型Belief及实际轨迹。Polygon投影不等同人工视觉分割。</p><a href="REPORT_zh.md">中文报告</a><pre>'+html.escape(json.dumps(result['summary'], ensure_ascii=False, indent=2))+'</pre>'+''.join(gallery))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    ax[0].bar(METHODS, [summary[m]['SR']*100 for m in METHODS]); ax[0].set_ylabel('SR (%)'); ax[0].set_ylim(0, 100)
    ax[1].bar(METHODS, [summary[m]['NE'] for m in METHODS]); ax[1].set_ylabel('Polygon NE (m)')
    fig.savefig(a.output/'summary.png', dpi=140); plt.close(fig)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
