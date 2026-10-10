"""Matched closed-loop evaluation: current pixels and public map inputs only.

Private targets/reference paths are accessed only after rollout for scoring.
The reverse start is the measured forward endpoint, not an oracle goal pose.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import cv2
import numpy as np
import torch

from .data import DEFAULT_DATA, DEFAULT_SBF
from .model import CoreModel
from .vision import FrozenROIEncoder
from graphnav.a3_runtime import A3WaypointPolicy
from graphnav.checkpointing import file_sha256
from graphnav.closedloop import PublicEpisode, ClosedLoopProtocol, rollout, score_trace, aggregate
from graphnav.observed_landmarks import LandmarkSpatialIndex
from graphnav.semantic_geometry import valid_surface_polygon
from graphnav.trajectory_policy import calibrated_stop_probability
from graphnav.joint_inference_controls import joint_stop_decision

DEFAULT_A3 = '/home/rental/20260922_1/Workspace/2027-CVPR/GraphNav/runs/static_final_analysis_20261010/E1_seed1/best.pt'


def public_landmarks(objects):
    """All usable public polygons, including unnamed target objects."""
    result = []
    for oid, item in objects.items():
        polygon = np.asarray(item.get('contour', []), dtype=np.float32)
        if not valid_surface_polygon(polygon):
            continue
        result.append(dict(id=str(oid), name=str(item.get('name', '')),
            object_type=str(item.get('object_type', 'Unknown')), contour=polygon,
            low=polygon.min(0), high=polygon.max(0)))
    return LandmarkSpatialIndex(result)


def public_episode(record, source, catalog):
    if record['direction'] not in ('forward', 'reverse'):
        raise ValueError('unsupported trajectory direction')
    original = source[record['source_episode_index']]
    if original.map_name != record['map_name'] or original.split != record['split']:
        raise ValueError('episode source index/map/split mismatch')
    if record['direction'] == 'forward':
        if original.instruction != record['instruction']:
            raise ValueError('forward source instruction mismatch')
        names = tuple(catalog.referenced_landmark_names(original))
    else:
        description = record.get('reverse_geometry_description') or {}
        anchor = description.get('public_anchor') or {}
        names = (anchor['name'],) if anchor.get('name') else ()
    # Language/map bindings only; never reverse destination coordinates.
    return PublicEpisode(record['map_name'], record['instruction'],
        np.asarray(record['trajectory_xyzyaw'][0], dtype=np.float64), (), names)


class MapBuilder:
    def __init__(self, root):
        self.root = Path(root)
        self.transforms = {}

    def transform(self, name):
        if name not in self.transforms:
            from sbfnav.coordinates import MapTransform
            self.transforms[name] = MapTransform.from_raster(self.root/'rgbd'/f'{name}.tif')
        return self.transforms[name]


class CurrentBuilder:
    """No nonpositive-height mirrored camera crops; those views remain Unknown."""
    def __init__(self, maps, directory):
        from graphnav.renderer import CityNavObservedViewRenderer
        self.maps = maps
        self.renderer = CityNavObservedViewRenderer(maps.root/'rgbd', directory, image_size=384)

    def transform(self, name):
        return self.maps.transform(name)

    def build_from_poses(self, episode, poses):
        from sbfnav.mapdata import GROUND_LEVEL
        pose = np.asarray(poses[-1], dtype=np.float64).copy()
        valid = bool(pose[2] > GROUND_LEVEL[episode.map_name])
        observation = self.renderer.current_rgbd(episode.map_name, pose, GROUND_LEVEL[episode.map_name]) if valid else None
        if observation is not None:
            observation.pop('surface_height_m', None)
        return SimpleNamespace(pose=pose, current_observation=observation, step=len(poses)-1)


class CorePolicy(A3WaypointPolicy):
    def __init__(self, checkpoint, head_checkpoint, directory, encoder, device='cuda', static_cache=None):
        super().__init__(checkpoint, directory, device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Our own checkpoint contains RNG state, hence weights_only=False.
        payload = torch.load(head_checkpoint, map_location='cpu', weights_only=False)
        self.protocol = payload['protocol']
        if self.protocol['frozen_resources'].get(str(Path(checkpoint).resolve())) != file_sha256(checkpoint):
            raise ValueError('head/static frozen checkpoint mismatch')
        self.head = CoreModel(hidden=self.protocol['hidden'], top_k=self.protocol['top_k']).to(self.device).eval()
        self.head.load_state_dict(payload['model'], strict=True)
        self.visual = encoder
        self.landmark_cache = {}
        self.exact_static = {}
        self.require_exact_static = static_cache is not None
        if static_cache is not None:
            root = Path(static_cache)
            manifest = json.loads((root/'manifest.json').read_text())
            if not manifest['complete'] or manifest['frozen_resources'].get(str(Path(checkpoint).resolve())) != file_sha256(checkpoint):
                raise ValueError('static query cache is incomplete or uses another A3')
            if file_sha256(root/'queries.pt') != manifest['files']['queries.pt']:
                raise ValueError('static query cache changed')
            if file_sha256(root/'queries.jsonl') != manifest['files'].get('queries.jsonl'):
                raise ValueError('static public query records changed or not fingerprinted')
            records = [json.loads(line) for line in (root/'queries.jsonl').read_text().splitlines()]
            tensors = torch.load(root/'queries.pt', map_location='cpu', weights_only=True)
            if len(records) != len(tensors):
                raise ValueError('static cache query/tensor alignment mismatch')
            self.exact_static = {json.dumps(record, sort_keys=True):tensor for record, tensor in zip(records,tensors)}
            if not records or len(self.exact_static) != len(records):
                raise ValueError('empty or duplicated public static query cache')

    def set_episode(self, episode, transform, catalog):
        super().set_episode(episode, transform, catalog)
        key = json.dumps(self.record, sort_keys=True)
        if getattr(self, 'require_exact_static', False):
            if key not in self.exact_static:
                raise ValueError('public query missing from declared exact static cache')
            self.current_static = self.exact_static[key]
            self.native_text = self.current_static['native_text'][None].float().to(self.device)
        else:
            self.current_static = None
            self.native_text = self.visual.texts([episode.instruction]).float().to(self.device)
        if episode.map_name not in self.landmark_cache:
            self.landmark_cache[episode.map_name] = public_landmarks(catalog.objects[episode.map_name])
        self.landmarks = self.landmark_cache[episode.map_name]

    def frozen_field(self):
        if getattr(self, 'current_static', None) is None:
            return super().frozen_field()
        query = self.current_static
        field = SimpleNamespace(belief_logits=query['static_logits'][None].to(self.device),
            dense_context=query['static_context'][None].to(self.device))
        return field, {'grid_xy':query['grid_xy'][None].to(self.device)}

    @torch.inference_mode()
    def predict(self, state, instruction):
        from sbfnav.mapdata import GROUND_LEVEL
        field, inputs = self.frozen_field()
        regions, vectors = ([], torch.zeros(0, 1024))
        if state.current_observation is not None:
            regions, vectors = self.visual.encode([state.current_observation], [self.landmarks])[0]
        grid = inputs['grid_xy'].float()
        bounds = torch.tensor(self.record['world_bounds'], device=self.device, dtype=torch.float32)[None]
        centers = torch.tensor([r['center_world'] for r in regions], device=self.device, dtype=torch.float32).reshape(-1, 2)
        world = bounds[:, None, :2]+grid*(bounds[:, None, 2:]-bounds[:, None, :2])
        ids = torch.cdist(centers[None], world).argmin(-1) if len(regions) else torch.zeros(1, 0, device=self.device, dtype=torch.long)
        k = max(1, len(regions))
        features = torch.zeros(1, k, 1024, device=self.device)
        valid = torch.zeros(1, k, dtype=torch.bool, device=self.device)
        quality = torch.zeros(1, k, 3, device=self.device)
        grid_ids = torch.zeros(1, k, dtype=torch.long, device=self.device)
        if regions:
            features[0, :len(regions)] = vectors.float().to(self.device)
            valid[0, :len(regions)] = True
            quality[0, :len(regions)] = torch.tensor([r['quality'] for r in regions], device=self.device)
            grid_ids[0, :len(regions)] = ids[0]
        public = dict(static_logits=field.belief_logits.float(), static_context=field.dense_context.half().float(),
            grid_xy=grid, bounds=bounds, pose_world=torch.tensor(state.pose, device=self.device, dtype=torch.float32)[None],
            ground_level=torch.tensor([GROUND_LEVEL[self.record['map_name']]], device=self.device, dtype=torch.float32),
            native_text=self.native_text, roi_features=features, roi_grid_ids=grid_ids, roi_valid=valid, roi_quality=quality)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = self.head(public)
        candidate, mode, waypoint, stop = result.trajectory.select()
        chosen = result.candidate_ids[0, int(candidate[0])]
        predicted_goal = world[0, chosen].float().cpu().numpy()
        probability = float(calibrated_stop_probability(stop.float(), self.protocol['stop_positive_weight'])[0])
        should_stop, diagnostics = joint_stop_decision(pose_world=state.pose, candidate_world_xy=predicted_goal,
            stop_probability=probability, ground_level=GROUND_LEVEL[self.record['map_name']], threshold=.5, guard_radius_m=20.)
        diagnostics.update(observed_rois=len(regions), predicted_goal_xy=predicted_goal.tolist(),
            top1_cell=int(result.belief_logits[0].argmax()), static_top1_cell=int(field.belief_logits[0].argmax()),
            candidate=int(candidate[0]), mode=int(mode[0]))
        return SimpleNamespace(xyz=waypoint[0].float().cpu().numpy(), should_stop=should_stop, diagnostics=diagnostics)


def run(args):
    sys.path.insert(0, args.sbf_source)
    from sbfnav.dataset import CityNavDataset, CityReferCatalog
    from sbfnav.mapdata import GROUND_LEVEL
    from sbfnav.navigation import BoundedWaypointController
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fixed_inputs = dict(split=args.split, episodes_sha256=file_sha256(args.episodes),
        checkpoint_sha256=file_sha256(args.checkpoint), a3_sha256=file_sha256(args.a3), limit=args.limit,
        horizon=20, horizontal_step_m=50., vertical_step_m=10., stop_threshold=.5, stop_guard_m=20.,
        vision_revision='1b426889ea62b5a72bf9839009a1b184bfc9c178', max_rois=32,
        runtime_sha256=file_sha256(__file__), vision_code_sha256=file_sha256(Path(__file__).with_name('vision.py')))
    fixed_inputs.update(data_root=str(Path(args.data_root).resolve()), device=args.device,
        public_objects_sha256=file_sha256(Path(args.data_root)/'cityrefer/objects.json'),
        public_language_bindings_sha256=file_sha256(Path(args.data_root)/'cityrefer/processed_descriptions.json'))
    static_cache = getattr(args, 'static_cache', None)
    fixed_inputs['static_cache_manifest_sha256'] = file_sha256(Path(static_cache)/'manifest.json') if static_cache else None
    if (output/'predictions.jsonl').exists() and not (output/'protocol.json').exists():
        raise ValueError('predictions without an immutable evaluation protocol')
    if (output/'protocol.json').exists():
        previous = json.loads((output/'protocol.json').read_text())
        if any(previous.get(k) != v for k, v in fixed_inputs.items()):
            raise ValueError('refusing to resume predictions under changed evaluation inputs')
    records = [json.loads(line) for line in Path(args.episodes).read_text().splitlines()]
    if any(r['split'] != args.split for r in records):
        raise ValueError('episode file split differs from declared evaluation split')
    if args.limit:
        families = sorted({r['family_id'] for r in records})[:args.limit]
        records = [r for r in records if r['family_id'] in families]
    # Map order changes only loading efficiency, never sample inclusion.
    records.sort(key=lambda r:(r['map_name'], r['family_id'], r['direction']))
    source = CityNavDataset(args.data_root, split=args.split, version='processed_citynav')
    catalog = CityReferCatalog(Path(args.data_root)/'cityrefer')
    builder = CurrentBuilder(MapBuilder(args.data_root), output/'renderer')
    encoder = FrozenROIEncoder(device=args.device)
    policy = CorePolicy(args.a3, args.checkpoint, output/'static_query', encoder, device=args.device, static_cache=static_cache)
    protocol = ClosedLoopProtocol()
    controller = BoundedWaypointController(horizontal_step_m=50., vertical_step_m=10.)
    manifest = dict(**fixed_inputs, current_pixels_only=True, no_target_in_policy=True,
        condition=policy.protocol['condition'], evaluation_selection='all predetermined pilot families',
        start_seconds=time.time())
    (output/'protocol.json').write_text(json.dumps(manifest, indent=2))
    prediction_path = output/'predictions.jsonl'
    done = {}
    if prediction_path.exists():
        previous_rows = list(map(json.loads, prediction_path.read_text().splitlines()))
        done = {r['episode_id']:r for r in previous_rows}
        selected = {r['episode_id'] for r in records}
        if len(done) != len(previous_rows) or not set(done) <= selected:
            raise ValueError('duplicate or foreign episode predictions in resumed evaluation')
    with prediction_path.open('a') as handle:
        for index, record in enumerate(records):
            if record['episode_id'] in done:
                continue
            public = public_episode(record, source, catalog)
            policy.set_episode(public, builder.transform(public.map_name), catalog)
            poses, termination, timing, diagnostics = rollout(public, builder, controller, policy.predict,
                GROUND_LEVEL[public.map_name], protocol, synchronize=lambda:torch.cuda.synchronize())
            # Only now access private evaluation coordinates.
            metrics = score_trace(poses, record['private_labels']['target_world'], record['trajectory_xyzyaw'], termination,
                radius=20., decision_seconds=timing)
            result = dict(episode_id=record['episode_id'], family_id=record['family_id'], map_name=record['map_name'],
                direction=record['direction'], static_language_eligible=record['static_language_eligible'],
                metrics=metrics, poses=poses.tolist(), termination=termination, diagnostics=diagnostics)
            handle.write(json.dumps(result)+'\n'); handle.flush(); done[record['episode_id']] = result
            if (index+1)%10 == 0:
                print(json.dumps(dict(done=len(done), total=len(records), condition=policy.protocol['condition'])), flush=True)
    rows = list(done.values())
    summary = {direction:aggregate([r['metrics'] for r in rows if r['direction']==direction]) for direction in ('forward','reverse')}
    for eligible, name in ((True, 'reverse_geometry_language'), (False, 'reverse_trajectory_only')):
        group = [r['metrics'] for r in rows if r['direction']=='reverse' and r['static_language_eligible']==eligible]
        summary[name] = aggregate(group) if group else {'episodes':0}
    summary['completed_episodes'] = len(rows)
    summary['eligible_reverse_episodes'] = sum(r['direction']=='reverse' and r['static_language_eligible'] for r in rows)
    summary['elapsed_seconds'] = time.time()-manifest['start_seconds']
    (output/'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='val_seen', choices=['val_seen','val_unseen'])
    parser.add_argument('--a3', default=DEFAULT_A3)
    parser.add_argument('--data-root', default=DEFAULT_DATA)
    parser.add_argument('--sbf-source', default=DEFAULT_SBF)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--static-cache', help='exact frozen public query fields from the Seen cache')
    parser.add_argument('--limit', type=int, default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
