"""HETT-style episode profiling, without synchronized model-phase wrappers.

CUDA Event module spans include host dispatch gaps and are not kernel-only
measurements. Observation timing retains HETT's synchronized host wall clock.
The shared GraphNav rollout's normal per-decision synchronization is unchanged.
"""
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import time
from .profiling import DecisionProfiler


GPU_KEYS = ('language_gpu_ms', 'vision_gpu_ms', 'static_graph_gpu_ms',
            'visual_residual_history_gpu_ms', 'trajectory_stop_gpu_ms')


class HETTEpisodeProfiler(DecisionProfiler):
    def instrument(self, policy, builder):
        self.wrap(builder, 'build_from_poses', 'environment_ms')
        self.gpu_enabled = True
        for module, name in ((policy.bert.model, 'language_gpu_ms'),
                             (policy.native.model.text_model, 'language_gpu_ms'),
                             (policy.native.model.vision_model, 'vision_gpu_ms'),
                             (policy.model.a3, 'static_graph_gpu_ms'),
                             (policy.model, 'visual_residual_history_gpu_ms'),
                             (policy.model.trajectory, 'trajectory_stop_gpu_ms')):
            self.gpu_hook(module, name)
        from . import observed_patch_scales
        original = observed_patch_scales.masked_native_pool

        def pooled(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            self.gpu_begin('vision_gpu_ms')
            try:
                return original(*args, **kwargs)
            finally:
                self.gpu_end()
        observed_patch_scales.masked_native_pool = pooled

    @contextmanager
    def sample(self, map_name, step):
        self.synchronize()
        started = time.perf_counter()
        self.parts = defaultdict(float)
        self.gpu_intervals = []
        self.active = True
        try:
            yield
        finally:
            self.synchronize()
            total = (time.perf_counter()-started)*1000
            self.active = False
            gpu = defaultdict(float)
            for interval in self.gpu_intervals:
                elapsed = interval['start'].elapsed_time(interval['end'])
                children = sum(child['start'].elapsed_time(child['end']) for child in interval['children'])
                gpu[interval['name']] += max(0., elapsed-children)
            components = {key: gpu[key] for key in GPU_KEYS}
            model = sum(components.values())
            environment = self.parts['environment_ms']
            self.rows.append(dict(map_name=map_name, episode_id=self.episode_id, step=step,
                                  rollout_ms=total, **components, model_inference_gpu_ms=model,
                                  environment_ms=environment, other_rollout_ms=max(0., total-model-environment)))

    def save(self, directory):
        super().save(directory)
        directory = Path(directory)
        timing = ('HETT-style module CUDA Event spans, nested child spans subtracted to avoid double counting; '
                  'no synchronized host wrappers around model modules. Environment observations use synchronized '
                  'host wall time. Normal shared-rollout per-decision synchronization remains. Episode totals '
                  'include public episode setup through final decision; model loading and post-rollout metrics '
                  'are excluded. CUDA intervals may include host launch gaps; vision includes native ROI pooling.')
        for name in ('profile.json', 'profile_episodes.json'):
            p = directory/name
            report = json.loads(p.read_text())
            report['timing'] = timing
            report['environment'] = 'See launch_record.json and gpu_activity.jsonl for observed concurrent GPU processes.'
            p.write_text(json.dumps(report, indent=2)+'\n')
