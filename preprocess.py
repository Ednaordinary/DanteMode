import os

import torch
from diffusers import FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, quantize, qint8

# For preprocessing any weights

dtype = torch.bfloat16

text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=dtype,
                                                revision="refs/pr/3")
transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer",
                                                                     torch_dtype=dtype, revision="refs/pr/3")
quantize(transformer, weights=qint8)
freeze(transformer)
quantize(text_encoder_2, weights=qint8)
freeze(text_encoder_2)
os.makedirs("models/flux-d/transformer", exist_ok=True)
transformer.save_pretrained("models/flux-d/transformer")
os.makedirs("models/flux-d/text_encoder_2", exist_ok=True)
text_encoder_2.save_pretrained("models/flux-d/text_encoder_2")
