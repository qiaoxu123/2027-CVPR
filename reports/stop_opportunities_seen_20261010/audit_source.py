"""Private post-rollout Stop diagnostics; never an inference input or policy."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from graphnav.closedloop import ClosedLoopProtocol
from graphnav.checkpointing import provenance
from scripts.summarize_closedloop import read_run


def classify(row, target, ground, protocol, threshold):
    poses = np.asarray(row['poses'], dtype=np.float64)
    observed = len(row['decision_seconds'])
    if not 0 < observed <= len(poses):
        raise ValueError('invalid observed decision count')
    near = np.linalg.norm(poses[:, :2] - np.asarray(target)[:2], axis=-1) <= protocol.success_radius_m
    success = bool(near[-1])
    if success != bool(row['metrics']['SR']) or bool(near.any()) != bool(row['metrics']['OSR']):
        raise ValueError('private target or metric protocol mismatch')
    # A horizon endpoint can be executed without another observation/decision.
    near_observed = near[:observed]
    height = poses[:observed, 2] - ground
    legal = (height >= protocol.min_altitude_m - 1e-5) & (height <= protocol.max_altitude_m + 1e-5)
    legal_near = near_observed & legal
    diagnostics_available = 'step_diagnostics' in row
    diagnostics = row.get('step_diagnostics', [{} for _ in range(observed)])
    if len(diagnostics) != observed:
        raise ValueError('decision diagnostics count mismatch')
    head_frames = [diagnostics[j] for j in np.flatnonzero(legal_near)
                   if 'stop_probability' in diagnostics[j] and 'predicted_goal_distance' in diagnostics[j]]
    return dict(success=success, failed=not success, decision_diagnostics_available=diagnostics_available,
                observed_target_near_frames=int(near_observed.sum()),
                observed_legal_arrival_frames=int(legal_near.sum()),
                failed_after_observed_legal_arrival=bool(not success and legal_near.any()),
                failed_after_only_illegal_observed_arrival=bool(not success and near_observed.any() and not legal_near.any()),
                failed_without_observed_arrival=bool(not success and not near_observed.any()),
                failed_policy_stop=bool(not success and row['termination'] == 'policy_stop'),
                legal_arrival_head_frames=len(head_frames),
                max_stop_probability_at_legal_arrival=max((d['stop_probability'] for d in head_frames), default=None),
                probability_rejection_frames=sum(d['stop_probability'] < threshold for d in head_frames),
                predicted_goal_gate_rejection_frames=sum(d['predicted_goal_distance'] > 1e-3 for d in head_frames))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='name=completed rollout directory')
    for name in ('sbf-source', 'data-root', 'split', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('refusing Stop diagnostic overwrite')
    sys.path.insert(0, args.sbf_source)
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    dataset = CityNavDataset(args.data_root, args.split, version='processed_citynav')
    models, rows = {}, []
    for spec in args.run:
        name, path = spec.split('=', 1)
        if name in models:
            raise ValueError('duplicate model')
        report, traces, manifest = read_run(path)
        if report['split'] != args.split:
            raise ValueError('run split mismatch')
        protocol = ClosedLoopProtocol(**report['protocol'])
        threshold = report['args'].get('stop_threshold', .3)
        model_rows = []
        for index, trace in traces.items():
            source = dataset[index]
            if list(source.episode_id) != trace['episode_id'] or not np.allclose(source.start_pose, trace['poses'][0], atol=1e-9, rtol=0):
                raise ValueError('private source Episode/start mismatch')
            result = classify(trace, source.target_position, GROUND_LEVEL[source.map_name], protocol, threshold)
            model_rows.append(dict(model=name, index=index, map_name=source.map_name,
                                   termination=trace['termination'], NE_m=trace['metrics']['NE'], **result))
        rows.extend(model_rows)
        models[name] = dict(path=path, episodes=len(model_rows), manifest_sha256=manifest,
                           counts={key: sum(row[key] for row in model_rows) for key in
                                   ('failed', 'observed_target_near_frames', 'observed_legal_arrival_frames',
                                    'failed_after_observed_legal_arrival', 'failed_after_only_illegal_observed_arrival',
                                    'failed_without_observed_arrival', 'failed_policy_stop', 'legal_arrival_head_frames',
                                    'probability_rejection_frames', 'predicted_goal_gate_rejection_frames')})
    output.mkdir(parents=True)
    with (output / 'episodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result = dict(models=models, provenance=provenance(), note='Private post-rollout GT diagnostics only; no model selection/input or policy change. Observed poses are exactly the decision prefix, not an unobserved horizon endpoint. Public legal altitude is separate from unchanged XY success. A GT-near visit is an oracle diagnostic opportunity, not evidence the policy could identify the target. Head-frame rejection counts may overlap; absent head information is not Stop calibration evidence.')
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
