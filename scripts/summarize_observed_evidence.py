"""Summarize public, actually executed visual evidence; no target labels."""
import argparse
import csv
import json
from pathlib import Path

from graphnav.checkpointing import file_sha256
from scripts.summarize_closedloop import read_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='name=completed run')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('refusing evidence report overwrite')
    summaries, records = {}, []
    for spec in args.run:
        name, folder = spec.split('=', 1)
        if name in summaries:
            raise ValueError('duplicate model')
        report, episodes, manifest = read_run(folder)
        total, enabled, with_regions, with_refs, with_aux = 0, 0, 0, 0, 0
        enabled_episodes = 0
        for index, episode in episodes.items():
            diagnostics = episode.get('step_diagnostics', [])
            if len(diagnostics) != len(episode['decision_seconds']):
                raise ValueError('diagnostics must cover actual observed decisions')
            visual = [d for d in diagnostics if d.get('visual')]
            if not visual:
                raise ValueError('requested run has no enabled visual observations')
            required = ('observed_regions', 'observed_nodes', 'auxiliary_nodes', 'observed_candidate_fraction')
            if any(any(k not in d for k in required) for d in visual):
                raise ValueError('missing actual evidence telemetry')
            histories = {bool(d.get('history')) for d in visual}
            if len(histories) != 1:
                raise ValueError('history mode changed within episode')
            regions = sum(d['observed_regions'] > 0 for d in visual)
            refs = sum(d['observed_nodes'] > 0 for d in visual)
            aux = sum(d['auxiliary_nodes'] > 0 for d in visual)
            records.append(dict(model=name, index=index, map_name=episode['map_name'],
                decisions=len(diagnostics), visual_decisions=len(visual), history=histories.pop(),
                decisions_with_regions=regions, decisions_with_referenced_nodes=refs,
                decisions_with_auxiliary_nodes=aux, final_observed_grid_fraction=visual[-1]['observed_candidate_fraction']))
            total += len(diagnostics)
            enabled += len(visual)
            with_regions += regions
            with_refs += refs
            with_aux += aux
            enabled_episodes += bool(regions)
        own = [r for r in records if r['model'] == name]
        summaries[name] = dict(path=folder, episodes=len(episodes), decisions=total,
            visual_decisions=enabled, decision_region_coverage=with_regions / enabled,
            decision_referenced_node_coverage=with_refs / enabled,
            decision_auxiliary_node_coverage=with_aux / enabled,
            episode_any_region_coverage=enabled_episodes / len(episodes),
            mean_final_observed_grid_fraction=sum(r['final_observed_grid_fraction'] for r in own) / len(own),
            episode_manifest_sha256=manifest,
            predictions_sha256=file_sha256(Path(folder) / 'predictions.jsonl'),
            observed_visual_cache=report.get('observed_visual_cache'))
    output.mkdir(parents=True)
    result = dict(models=summaries, script_sha256=file_sha256(__file__),
        note='Public policy telemetry only. No target coordinates, target IDs, expert paths or future frames are read. Current-only runs count current trusted projected regions; history runs count accumulated trusted regions and are not directly comparable as instantaneous coverage. Referenced-node coverage means an instruction-referenced public landmark has visual evidence, not that the goal is visible or recognized. Observed-grid coverage is cumulative geometric FOV, not semantic visibility, navigable-space coverage or accuracy. Unobserved endpoints after the final decision are never counted.')
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    with (output / 'episodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
