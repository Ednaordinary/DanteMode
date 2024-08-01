import os

import torch
from diffusers import FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, quantize, qint8
from optimum.quanto.models.diffusers_models import QuantizedDiffusersModel
from optimum.quanto.models.transformers_models import QuantizedModelForCausalLM

# For preprocessing any weights

dtype = torch.bfloat16

class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):

    base_class = FluxTransformer2DModel

print("Loading text encoder")
text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=dtype,
                                                revision="refs/pr/3")
print("Loading transformer")
transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer",
                                                                     torch_dtype=dtype, revision="refs/pr/3")
print("Quantizing Transformer")
transformer = QuantizedFluxTransformer2DModel.quantize(transformer, qint8)
print("Saving Transformer")
os.makedirs("models/flux-d/transformer", exist_ok=True)
transformer.save_pretrained("models/flux-d/transformer")

print("Quantizing text encoder")
text_encoder_2 = QuantizedModelForCausalLM.quantize(text_encoder_2, qint8)
print("Saving text encoder")
os.makedirs("models/flux-d/text_encoder_2", exist_ok=True)
text_encoder_2.save_pretrained("models/flux-d/text_encoder_2")
