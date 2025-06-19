import torch
import numpy as np

from ..utils import np_save

class EmbedWrapper:
    def __init__(self):
        self.cls = None
        self.init_args = {}
        self.call_name = "encode_prompt"
        self.allowed_args = ["prompt"]


    def embeds(self, **kwargs) -> torch.Tensor:
        pipe = self.cls(**self.init_args)
        pipe.to("cuda")
        return pipe.__dict__[self.call_name](**kwargs)

    def convert_npy(self, embeds: torch.Tensor) -> np.ndarray:
        embeds = embeds.float().numpy(force=True)
        return embeds