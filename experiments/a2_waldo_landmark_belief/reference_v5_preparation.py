"""CPU admission and weighted CUDA smoke for the isolated V5 run."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

from .paths import ROOT, sha, save
from .reference_overlay_context import BASE
from .reference_v5_train import _json_sha, source_binding


PREPARATION = BASE / "revision_v5"
TEST_FILES = (
    "tests/test_a2_reference_v5_weighting.py",
    "tests/test_a2_reference_v5_data.py",
    "tests/test_a2_reference_v5_protocol.py",
    "tests/test_a2_reference_v5_train.py",
    "tests/test_a2_reference_v5_preparation.py",
    "tests/test_a2_reference_v5_verify.py",
)


def test_hashes():
    return {name: sha(ROOT / name) for name in TEST_FILES}


def proof_binding_sha(binding):
    return _json_sha(dict(binding=binding, test_source_sha256=test_hashes()))


def assert_cpu_environment(*, visible_devices=None, cuda_initialized=False):
    if visible_devices not in (None, "") or cuda_initialized:
        raise ValueError("V5 CPU preparation must not see or initialize CUDA")


def validate_pretraining_proofs(binding, directory=PREPARATION, *, expected_sha256=None):
    directory = Path(directory)
    calculated = proof_binding_sha(binding)
    if expected_sha256 is not None and expected_sha256 != calculated:
        raise ValueError("V5 source binding differs from the expected training proof")
    expected = expected_sha256 or calculated
    cpu_path, log_path = directory / "cpu_gates.json", directory / "cpu_tests.log"
    cuda_path = directory / "cuda_smoke.json"
    validate_cpu_proof(binding, directory, expected_sha256=expected)
    cuda = json.loads(cuda_path.read_text())
    if (cuda.get("status") != "v5_weighted_TRAIN_CUDA_backward_passed"
            or cuda.get("binding_sha256") != expected
            or cuda.get("rank_backward_steps") != 3
            or cuda.get("refine_backward_steps") != 3
            or cuda.get("all_finite") is not True
            or cuda.get("frozen_parameters_unchanged") is not True
            or cuda.get("weighted_rows") is not True
            or cuda.get("benchmark_used") is not False):
        raise ValueError("V5 weighted CUDA smoke proof is incomplete or foreign")


def validate_cpu_proof(binding, directory=PREPARATION, *, expected_sha256=None):
    directory = Path(directory)
    calculated = proof_binding_sha(binding)
    if expected_sha256 is not None and expected_sha256 != calculated:
        raise ValueError("V5 CPU source binding differs from the expected proof")
    expected = expected_sha256 or calculated
    cpu_path, log_path = directory / "cpu_gates.json", directory / "cpu_tests.log"
    cpu = json.loads(cpu_path.read_text())
    text = log_path.read_text()
    summary = cpu.get("unit_test_summary", "")
    if (cpu.get("status") != "v5_cpu_tests_and_weighting_verified"
            or cpu.get("binding_sha256") != expected
            or cpu.get("all_cpu_tests_passed") is not True
            or cpu.get("no_gpu_used") is not True
            or cpu.get("returncode") != 0
            or cpu.get("test_log_sha256") != sha(log_path)
            or re.match(r"[1-9][0-9]* passed", summary) is None
            or summary not in text):
        raise ValueError("V5 CPU test proof is incomplete or belongs to another source")


def prepare_cpu(directory=PREPARATION):
    assert_cpu_environment(visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                           cuda_initialized=torch.cuda.is_initialized())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binding = source_binding()
    bound = proof_binding_sha(binding)
    log = directory / "cpu_tests.log"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *TEST_FILES],
        cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.write_text(result.stdout)
    match = re.search(r"[1-9][0-9]* passed[^\n]*", result.stdout)
    if result.returncode or not match or proof_binding_sha(source_binding()) != bound:
        raise ValueError("V5 CPU tests failed or source binding changed")
    save(dict(status="v5_cpu_tests_and_weighting_verified", binding_sha256=bound,
              test_source_sha256=test_hashes(), test_log_sha256=sha(log),
              unit_test_summary=match.group(), returncode=0,
              all_cpu_tests_passed=True, no_gpu_used=True), directory / "cpu_gates.json")
    validate_cpu_proof(binding, directory, expected_sha256=bound)


def cuda_smoke(directory=PREPARATION):
    from .batching import collate_views, to_device
    from .reference_v4_train import training_signature as v4_training_signature
    from .reference_v5_data import TargetBalancedReferenceViews
    from .relation_v3_loss import relation_v3_loss
    from .relation_v3_train import fingerprint, initial_model, optimizer_for, require_idle_gpu

    directory = Path(directory)
    binding = source_binding()
    bound = proof_binding_sha(binding)
    validate_cpu_proof(binding, directory, expected_sha256=bound)
    require_idle_gpu()
    dataset = TargetBalancedReferenceViews("train")
    selected, families = [], {}
    for index, entry in enumerate(dataset.entries):
        family = entry["source_kind"]
        if families.get(family, 0) >= 2:
            continue
        row = dataset[index]
        labels = row["labels"]
        positive = labels["localization_utility"] & labels["utility_valid"]
        negative = ~positive & labels["utility_valid"]
        if positive.any() and negative.any():
            selected.append(row)
            families[family] = families.get(family, 0) + 1
        if len(selected) == 6:
            break
    if len(selected) != 6 or sorted(families.values()) != [2, 2, 2]:
        raise ValueError("V5 smoke requires two actual weighted TRAIN rows from each source")

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = initial_model()
    if fingerprint(model.state_dict()) != v4_training_signature()["initial_weights_sha256"]:
        raise ValueError("V5 smoke initial weights differ from frozen V4 parent")
    frozen = {name: param.detach().cpu().clone()
              for name, param in model.named_parameters() if not param.requires_grad}
    require_idle_gpu()
    model = model.cuda().train()
    batch = to_device(collate_views(selected), "cuda")
    optimizer = optimizer_for(model)
    records = []
    for step in range(6):
        require_idle_gpu(allow_self=True)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = relation_v3_loss(model, model(**batch["network"]), batch, refine=step >= 3)
        if not torch.isfinite(loss):
            raise ValueError("V5 weighted smoke produced a nonfinite loss")
        loss.backward()
        trainable = [param for param in model.parameters() if param.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
        if not norm > 0:
            raise ValueError("V5 weighted smoke produced no gradient")
        optimizer.step()
        records.append(dict(step=step, refine=step >= 3, loss=float(loss.detach()),
                            row_weights=batch["labels"]["sample_weight"].detach().cpu().tolist()))
    unchanged = all(torch.equal(dict(model.named_parameters())[name].detach().cpu(), value)
                    for name, value in frozen.items())
    if not unchanged or fingerprint(model.state_dict()) == v4_training_signature()["initial_weights_sha256"]:
        raise ValueError("V5 weighted smoke changed a frozen asset or failed to update")
    report = dict(status="v5_weighted_TRAIN_CUDA_backward_passed", binding_sha256=bound,
        source_families=families, rank_backward_steps=3, refine_backward_steps=3,
        all_finite=True, frozen_parameters_unchanged=True, weighted_rows=True,
        benchmark_used=False, steps=records)
    save(report, directory / "cuda_smoke.json")
    validate_pretraining_proofs(binding, directory, expected_sha256=bound)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-cpu", action="store_true")
    parser.add_argument("--cuda-smoke", action="store_true")
    args = parser.parse_args()
    if args.prepare_cpu == args.cuda_smoke:
        parser.error("choose exactly one of --prepare-cpu or --cuda-smoke")
    if args.prepare_cpu:
        prepare_cpu()
    else:
        import fcntl
        with (BASE / "gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            cuda_smoke()


if __name__ == "__main__":
    main()
