"""Native shared-space SigLIP2; patch and old identity heads stay separate."""
import torch
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder


class Encoder:
    def __init__(self, device='cuda', microbatch=16, **kwargs):
        self.native = ObservedSigLIPMatchingEncoder(device=device, microbatch=microbatch)
        self.dimension = self.native.dimension
        self.model = self.native.model
        self.metadata = dict(model='google/siglip2-large-patch16-384',
                             revision=getattr(self.model.config, '_commit_hash', None),
                             precision='bf16 autocast', input_size=384,
                             text_length=self.model.config.text_config.max_position_embeddings)

    def images(self, images):
        return self.native.images(images).float().cpu()

    def texts(self, texts):
        # Bound long instruction lists rather than allocating an entire split.
        return torch.cat([self.native.texts(texts[i:i+128]).float().cpu()
                          for i in range(0, len(texts), 128)])
