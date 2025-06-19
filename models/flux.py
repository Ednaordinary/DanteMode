from diffusers import FluxPipeline

from .generic import EmbedWrapper

class FluxEmbedWrapper(EmbedWrapper):
    def __init__(self):
        self.cls = FluxPipeline.from_pretrained
        self.init_args = {"pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev", "transformer": None, "vae": None}
        self.allowed_args = ["prompt"]
