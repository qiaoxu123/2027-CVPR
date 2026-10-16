"""Opt-in synchronized wall-clock component profiling for batch-one decisions."""
from collections import defaultdict
from contextlib import contextmanager
import csv
import json
import time
import numpy as np


class DecisionProfiler:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.rows = []
        self.active = False
        self.stack = []
        self.episode_id = None
        self.episodes=[]
        self.gpu_stack=[]
        self.gpu_intervals=[]
        self.gpu_enabled=False

    @contextmanager
    def phase(self, name):
        if not self.active:
            yield
            return
        self.synchronize()
        start = time.perf_counter()
        frame = [0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            self.stack.pop()
            self.parts[name] += elapsed - frame[0]
            if self.stack:
                self.stack[-1][0] += elapsed

    def wrap(self, obj, method, name):
        original = getattr(obj, method)
        def measured(*args, **kwargs):
            with self.phase(name):
                return original(*args, **kwargs)
        setattr(obj, method, measured)

    def start_episode(self, episode_id):
        self.synchronize()
        self.episode_started=time.perf_counter()
        self.episode_start_row=len(self.rows)
        self.episode_id=list(episode_id)

    def finish_episode(self):
        self.synchronize()
        elapsed=(time.perf_counter()-self.episode_started)*1000
        rows=self.rows[self.episode_start_row:]
        gpu_keys=('language_gpu_ms','vision_gpu_ms','static_graph_gpu_ms',
                  'visual_residual_history_gpu_ms','trajectory_stop_gpu_ms')
        components={key:sum(row.get(key,0.) for row in rows) for key in gpu_keys}
        gpu=sum(components.values());environment=sum(row['environment_ms'] for row in rows)
        self.episodes.append(dict(episode_id=self.episode_id,map_name=rows[0]['map_name'],
            decisions=len(rows),rollout_ms=elapsed,model_inference_gpu_ms=gpu,**components,
            environment_ms=environment,other_rollout_ms=max(0.,elapsed-gpu-environment)))

    def gpu_begin(self,name):
        import torch
        interval=dict(name=name,start=torch.cuda.Event(enable_timing=True),
                      end=torch.cuda.Event(enable_timing=True),children=[])
        interval['start'].record()
        if self.gpu_stack:self.gpu_stack[-1]['children'].append(interval)
        self.gpu_stack.append(interval)

    def gpu_end(self):
        interval=self.gpu_stack.pop()
        interval['end'].record()
        self.gpu_intervals.append(interval)

    def gpu_hook(self,module,name):
        def before(*unused):
            if self.active:self.gpu_begin(name)
        def after(*unused):
            if self.active:self.gpu_end()
        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)

    def instrument(self, policy, builder):
        self.wrap(builder, 'build_from_poses', 'environment_ms')
        self.wrap(policy.bert, 'encode_tokens', 'language_ms')
        self.wrap(policy.native, 'texts', 'language_ms')
        self.wrap(policy.visual, 'encode', 'vision_ms')
        self.wrap(policy.model.a3, 'forward', 'static_graph_ms')
        self.wrap(policy.model.trajectory, 'forward', 'trajectory_stop_ms')
        # Exclusive of nested trajectory/static calls: matching, history,
        # bounded residual, belief update and proposal selection.
        self.wrap(policy.model, 'forward', 'visual_residual_history_ms')
        self.gpu_enabled=True
        for module,name in ((policy.bert.model,'language_gpu_ms'),
                (policy.native.model.text_model,'language_gpu_ms'),
                (policy.native.model.vision_model,'vision_gpu_ms'),
                (policy.model.a3,'static_graph_gpu_ms'),
                (policy.model,'visual_residual_history_gpu_ms'),
                (policy.model.trajectory,'trajectory_stop_gpu_ms')):
            self.gpu_hook(module,name)
        # Native ROI pooling reuses patch K/V outside vision_model.forward.
        from . import observed_patch_scales
        original=observed_patch_scales.masked_native_pool
        def pooled(*args,**kwargs):
            if not self.active:return original(*args,**kwargs)
            self.gpu_begin('vision_gpu_ms')
            try:return original(*args,**kwargs)
            finally:self.gpu_end()
        observed_patch_scales.masked_native_pool=pooled


    @contextmanager
    def sample(self, map_name, step):
        self.synchronize()
        start = time.perf_counter()
        self.parts = defaultdict(float)
        self.gpu_intervals=[]
        self.active = True
        try:
            yield
        finally:
            self.synchronize()
            total = (time.perf_counter() - start) * 1000
            self.active = False
            gpu=defaultdict(float)
            if self.gpu_enabled:
                for interval in self.gpu_intervals:
                    elapsed=interval['start'].elapsed_time(interval['end'])
                    children=sum(child['start'].elapsed_time(child['end']) for child in interval['children'])
                    gpu[interval['name']]+=max(0.,elapsed-children)
                for key in ('language_gpu_ms','vision_gpu_ms','static_graph_gpu_ms','visual_residual_history_gpu_ms','trajectory_stop_gpu_ms'):gpu[key]+=0.
                gpu['model_inference_gpu_ms']=sum(gpu.values())
            categories = ('language_ms', 'vision_ms', 'static_graph_ms',
                          'visual_residual_history_ms', 'trajectory_stop_ms')
            model = sum(self.parts[k] for k in categories)
            self.rows.append(dict(map_name=map_name, episode_id=self.episode_id, step=step,
                rollout_ms=total, model_components_ms=model, **gpu,
                **{k:self.parts[k] for k in (*categories,'environment_ms')},
                other_rollout_ms=total-model-self.parts['environment_ms']))

    def save(self, directory):
        from pathlib import Path
        directory = Path(directory)
        if not self.rows:
            raise ValueError('No profiling samples')
        keys = [key for key in self.rows[0] if key.endswith('_ms')]
        def stats(rows):
            return {key:dict(min=float(np.min(values)),mean=float(np.mean(values)),
                population_variance=float(np.var(values))) for key in keys
                for values in [[row[key] for row in rows]]}
        report = dict(samples=len(self.rows), units='ms; population variance ms²',
            all=stats(self.rows), excluding_first=stats(self.rows[1:]) if len(self.rows)>1 else None,
            first=self.rows[0], slowest_five=sorted(self.rows,key=lambda r:r['rollout_ms'],reverse=True)[:5],
            timing='CUDA-synchronized host wall time, includes CPU processing/transfers within each phase; exclusive nested attribution. model_components_ms is a sum of model stages, not pure CUDA kernel time. Rollout sample is one observation/decision/controller step. Model loading excluded; first observation includes raster cold start. Synchronization adds overhead.',
            environment='HTNav training concurrently uses the same GPU; not an isolated HETT speed comparison.')
        if self.episodes:
            def episode_stats(rows):
                return {key:dict(min=float(np.min(values)),mean=float(np.mean(values)),population_variance=float(np.var(values))) for key in self.episodes[0] if key.endswith('_ms') for values in [[row[key] for row in rows]]}
            episode_report=dict(samples=len(self.episodes),units='ms; population variance ms²',all=episode_stats(self.episodes),excluding_first=episode_stats(self.episodes[1:]) if len(self.episodes)>1 else None,first=self.episodes[0],slowest_five=sorted(self.episodes,key=lambda r:r['rollout_ms'],reverse=True)[:5],samples_detail=self.episodes,timing='Entire episode wall time before public-state setup through final rollout decision; CUDA Event module spans with nested child subtraction, plus native ROI pooling events. Model loading excluded. Other overhead is total minus GPU component sum minus synchronized observation wall time, as in original HETT profiler. CUDA intervals may include host launch gaps. HTNav concurrent GPU training affects latency.')
            (directory/'profile_episodes.json').write_text(json.dumps(episode_report,indent=2))
        (directory/'profile.json').write_text(json.dumps(report,indent=2))
        with (directory/'profile_samples.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=self.rows[0].keys())
            writer.writeheader();writer.writerows(self.rows)
