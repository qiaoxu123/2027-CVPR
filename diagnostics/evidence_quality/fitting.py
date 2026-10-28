"""Fit and durably freeze both CPU probes before any DEV assessment."""
import json
import math
from pathlib import Path
import tempfile

import torch

from experiments.a2_waldo_landmark_belief.paths import sha
from .packets import load_packets
from .linear_fit import fit_linear, predict, training_partition
from .conditions import choose_condition


ARMS = dict(baseline=15, extended=22)


def _curves(directory, models):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    fig, axis = plt.subplots(figsize=(7, 4))
    with SummaryWriter(str(directory/'tensorboard')) as writer:
        for arm, model in models.items():
            history = model['history']
            axis.plot([r['epoch'] for r in history], [r['loss'] for r in history], label=arm)
            for row in history:
                writer.add_scalar(f'{arm}/fit_loss', row['loss'], row['epoch'])
    axis.set(xlabel='CPU probe epoch', ylabel='TRAIN fit loss')
    axis.legend(); fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(directory/f'fit_curves.{extension}')
    plt.close(fig)


def _load_frozen(directory, signature, features):
    manifest = json.loads((directory/'manifest.json').read_text())
    files = manifest.get('files_sha256', {})
    required = {'baseline.pt', 'extended.pt', 'fixed_choice.json', 'fit_curves.png', 'fit_curves.svg'}
    if (manifest.get('status') != 'fixed20_cpu_probes_and_train_conditions'
            or manifest.get('signature') != signature or not required <= set(files)
            or not any(name.startswith('tensorboard/events.') for name in files)):
        raise ValueError('Frozen CPU fit signature or required artifacts changed')
    for name, digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(directory/relative) != digest:
            raise ValueError('Frozen CPU model/choice/curve bytes changed')
    decision = json.loads((directory/'fixed_choice.json').read_text())
    if decision.get('signature') != signature or set(decision.get('choices', {})) != set(ARMS):
        raise ValueError('Frozen CPU condition source changed')
    models = {}
    for arm, width in ARMS.items():
        state = torch.load(directory/f'{arm}.pt', map_location='cpu', weights_only=False)
        if (state.get('width') != width or state.get('epochs') != 20 or state.get('seed') != 0
                or state.get('initialization') != 'zero'
                or [r['epoch'] for r in state['history']] != list(range(1,21))
                or not all(math.isfinite(r['loss']) for r in state['history'])):
            raise ValueError('Incomplete or foreign fixed20 CPU fitting')
        snapshots = state.get('snapshots', [])
        if ([r['epoch'] for r in snapshots] != list(range(1,21))
                or any(tuple(r['weight'].shape) != (3,width) or tuple(r['bias'].shape) != (3,)
                       or not torch.isfinite(r['weight']).all() or not torch.isfinite(r['bias']).all()
                       for r in snapshots)
                or not torch.equal(snapshots[-1]['weight'], state['weight'])
                or not torch.equal(snapshots[-1]['bias'], state['bias'])):
            raise ValueError('All20 actual epoch weights must be retained')
        predict(state, features)  # Validate normalization and actual parameter axes.
        models[arm] = state
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(directory/'tensorboard'), size_guidance={'scalars':0}).Reload()
    for arm, state in models.items():
        values = {e.step:e.value for e in events.Scalars(f'{arm}/fit_loss')}
        if set(values) != set(range(1,21)) or any(not math.isclose(values[r['epoch']], r['loss'], rel_tol=1e-6, abs_tol=1e-6) for r in state['history']):
            raise ValueError('CPU TensorBoard/history values changed')
    return dict(models=models, decision=decision, manifest_sha256=sha(directory/'manifest.json'))


def fit_and_freeze(directory, exported):
    """Only TRAIN packets enter either model fitting or condition selection."""
    directory = Path(directory)
    data = load_packets(directory/'packets', role='train',
                        expected_entries=exported['entries']['train'], binding=exported['binding'])
    signature = dict(binding=exported['binding'], train_manifest_sha256=data['manifest_sha256'])
    features = torch.from_numpy(data['arrays']['features'])
    destination = directory/'fit'
    if destination.exists():
        return _load_frozen(destination, signature, features)
    fit = training_partition(data['identities'])
    selected = (~fit).numpy()
    ids = [e for e, keep in zip(data['identities'], selected) if keep]
    labels = torch.from_numpy(data['arrays']['outcomes']).long()
    pending = Path(tempfile.mkdtemp(prefix='.fit-pending-', dir=directory))
    models, choices = {}, {}
    for arm, width in ARMS.items():
        model = fit_linear(features, labels, data['identities'], width=width)
        models[arm] = model
        scores = predict(model, features).numpy()
        choices[arm] = choose_condition(identities=ids, scores=scores[selected],
            available=data['arrays']['available'][selected], before=data['arrays']['before'][selected],
            after=data['arrays']['after'][selected])
        torch.save(model, pending/f'{arm}.pt')
    decision = dict(signature=signature, choices=choices, diagnostic_only=True,
                    dev_used_for_training_or_selection=False,
                    fit_views=int(fit.sum()), selection_views=int((~fit).sum()))
    (pending/'fixed_choice.json').write_text(json.dumps(decision, indent=2)+'\n')
    _curves(pending, models)
    files = {str(p.relative_to(pending)):sha(p) for p in sorted(pending.rglob('*')) if p.is_file()}
    manifest = dict(status='fixed20_cpu_probes_and_train_conditions', signature=signature, files_sha256=files)
    (pending/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    pending.rename(destination)
    return _load_frozen(destination, signature, features)
