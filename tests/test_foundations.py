from types import SimpleNamespace
from collections import OrderedDict

import numpy as np
import pytest
import torch

from graphnav.foundations import CandidateVisionCache, FrozenBERT, FrozenSigLIP2
from graphnav.token_language import FrozenBERTTokens


def test_frozen_feature_caches_roundtrip_and_reject_protocol_changes(tmp_path):
    tokens = FrozenBERTTokens.__new__(FrozenBERTTokens)
    tokens.name, tokens.revision, tokens.max_length = "bert-test", "revision", 512
    vector, mask = torch.randn(2, 8), torch.ones(2).bool()
    tokens.cache = OrderedDict(query=(vector, mask))
    path = tmp_path / "tokens.pt"
    tokens.save_cache(path)
    tokens.cache.clear()
    tokens.load_cache(path)
    assert torch.equal(tokens.cache["query"][0], vector)
    tokens.max_length = 96
    with pytest.raises(ValueError, match="protocol"):
        tokens.load_cache(path)
    vision = CandidateVisionCache()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    valid = np.ones((2, 2), dtype=bool)
    vision.put(image, valid, vector)
    path = tmp_path / "vision.pt"
    metadata = {"revision": "test", "precision": "bf16"}
    vision.save(path, metadata)
    restored = CandidateVisionCache()
    restored.load(path, metadata)
    assert torch.equal(restored.get(image, valid), vector)
    with pytest.raises(ValueError, match="protocol"):
        restored.load(path, {**metadata, "precision": "fp32"})


def test_empty_siglip2_batch_uses_real_vision_config_width():
    encoder = FrozenSigLIP2.__new__(FrozenSigLIP2)
    encoder.device = torch.device("cpu")
    encoder.model = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(hidden_size=768)))
    assert encoder.encode([]).shape == (0, 768)


def test_frozen_bert_cache_does_not_retain_all_token_states():
    encoder = FrozenBERT.__new__(FrozenBERT)
    encoder.device = torch.device("cpu")
    encoder.max_length = 96
    encoder.cache = {}

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return Batch(count=len(texts))

    class Model:
        def __call__(self, count):
            return SimpleNamespace(
                last_hidden_state=torch.arange(count * 96 * 16).reshape(count, 96, 16).float()
            )

    encoder.tokenizer = Tokenizer()
    encoder.model = Model()
    actual = encoder.encode(["Church", "Building"])
    assert torch.equal(actual[0], torch.arange(16).float())
    assert torch.equal(actual[1], torch.arange(96 * 16, 96 * 16 + 16).float())
    assert encoder.cache["Church"].untyped_storage().nbytes() == 2 * 16 * 4
