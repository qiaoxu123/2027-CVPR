"""Complete BERT sequence tokens for frozen token-level candidate attention.

The 768-d CLS alone is insufficient for east/near/between compositional
grounding. Cache frozen last_hidden_state and *attention_mask*, NEVER labels.
Cached tensors are regular detached CPU tensors (not inference tensors).
"""

from __future__ import annotations

from collections import OrderedDict
import torch


class FrozenBERTTokens:
    def __init__(
        self, name="bert-base-uncased", device="cuda",
        max_length=96, cache_limit=50000,
    ):
        from transformers import AutoModel, AutoTokenizer
        self.name = name
        self.device = torch.device(device)
        self.max_length = max_length
        self.cache_limit = cache_limit
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.model.requires_grad_(False)
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def save_cache(self, path):
        torch.save({"name": self.name, "revision": self.revision,
                    "max_length": self.max_length, "cache": dict(self.cache)}, path)

    def load_cache(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state["name"], state["revision"], state["max_length"]) != (
                self.name, self.revision, self.max_length):
            raise ValueError("Frozen BERT cache protocol differs")
        self.cache = OrderedDict(state["cache"])

    @torch.inference_mode()
    def _load(self, texts):
        unknown = list(dict.fromkeys(text for text in texts if text not in self.cache))
        # Bound embedding batch to prevent BERT from allocating O(batch x 96^2).
        for start in range(0, len(unknown), 64):
            chunk = unknown[start:start + 64]
            if not chunk:
                continue
            batch = self.tokenizer(
                chunk, truncation=True, padding=True,
                max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**batch).last_hidden_state
            for i, phrase in enumerate(chunk):
                length = int(batch["attention_mask"][i].sum().item())
                # clone() outside GPU inference state and copy to CPU.
                vec = outputs[i, :length].detach().to("cpu").clone()
                valid = batch["attention_mask"][i, :length].bool().cpu().clone()
                self.cache[phrase] = (vec, valid)
                self.cache.move_to_end(phrase)
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)

    def encode_tokens(self, texts):
        """Return [B,L,768] and boolean [B,L]; L<=max_length."""
        texts = list(texts)
        if not texts:
            raise ValueError("text batch must be nonempty")
        self._load(texts)
        parts = [self.cache[t] for t in texts]
        length = max(x.shape[0] for x, _ in parts)
        dim = parts[0][0].shape[-1]
        output = torch.zeros(len(parts), length, dim, dtype=torch.float32)
        mask = torch.zeros(len(parts), length, dtype=torch.bool)
        for i, (data, validity) in enumerate(parts):
            output[i, :len(data)] = data.float()
            mask[i, :len(validity)] = validity
            self.cache.move_to_end(texts[i])
        return output.to(self.device), mask.to(self.device)

    def encode(self, texts):
        return self.encode_tokens(texts)[0][:, 0].contiguous()
