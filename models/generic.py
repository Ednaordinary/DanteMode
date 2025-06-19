import torch
import numpy as np


class EmbedWrapper:
    def __init__(self):
        self.cls = None
        self.init_args = {}
        self.allowed_args = ["prompt"]

    def load(self):
        self.pipe = self.cls(**self.init_args)
        self.pipe.to("cuda")

    def embeds(self, **kwargs) -> torch.Tensor:
        return self.pipe.encode_prompt(**kwargs)

    def convert_npy(self, embeds: torch.Tensor) -> np.ndarray:
        embeds = embeds.float().numpy(force=True)
        return embeds
