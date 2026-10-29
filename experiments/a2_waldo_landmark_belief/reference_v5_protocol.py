"""V5 protocol: isolate target-group weighting from the V4 model/loss."""
from copy import deepcopy

from .reference_v4_protocol import CONFIG as V4_CONFIG, choose_stage, verify_history


CONFIG = deepcopy(V4_CONFIG)
CONFIG.update(
    version="a2_relation_v5_target_group_balanced",
    target_weighting="equal group mass within source; real:teacher=8:1",
)


def validate_training_config():
    for key, value in V4_CONFIG.items():
        if key != "version" and CONFIG.get(key) != value:
            raise ValueError("V5 may not change V4 model or training controls")
    if (CONFIG.get("version") != "a2_relation_v5_target_group_balanced"
            or CONFIG.get("target_weighting") != "equal group mass within source; real:teacher=8:1"):
        raise ValueError("V5 target-group weighting protocol changed")
