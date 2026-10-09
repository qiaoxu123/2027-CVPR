from types import SimpleNamespace

import torch

from graphnav.foundations import FrozenBERT


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
