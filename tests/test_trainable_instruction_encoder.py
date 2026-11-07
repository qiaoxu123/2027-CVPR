"""Check instruction gradients and immutable name fallback without downloads."""
from types import SimpleNamespace

import torch

from graphnav.token_language import FrozenBERTTokens
from graphnav.trainable_instruction_encoder import TrainableInstructionBERT


class Batch(dict):
    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_live_tokens_and_frozen_missing_names(monkeypatch):
    def initialize(self, *args, **kwargs):
        self.model = TinyEncoder().eval().requires_grad_(False)
        self.device = torch.device('cpu')
        self.max_length = 96
        self.cache = {}
        self.tokenizer = lambda texts, **kw: Batch(
            input_ids=torch.ones(len(texts), 3, dtype=torch.long),
            attention_mask=torch.ones(len(texts), 3, dtype=torch.long),
        )

    monkeypatch.setattr(FrozenBERTTokens, '__init__', initialize)
    encoder = TrainableInstructionBERT()
    initial = encoder.encode(['known']).clone()
    tokens, mask = encoder.encode_tokens(['instruction'])
    assert tokens.requires_grad and mask.all()
    optimizer = torch.optim.SGD(encoder.model.parameters(), lr=.1)
    tokens.sum().backward()
    assert encoder.model.embedding.weight.grad.abs().sum() > 0
    optimizer.step()
    updated, _ = encoder.encode_tokens(['instruction'])
    assert not torch.equal(tokens, updated)
    # An alias first encountered after training must use the original encoder.
    alias = encoder.encode(['new alias'])
    torch.testing.assert_close(initial, alias, rtol=0, atol=0)
    torch.testing.assert_close(initial, encoder.encode(['known']), rtol=0, atol=0)
    assert not alias.requires_grad
    assert all(p.grad is None for p in encoder.frozen_names.parameters())
