import torch
from diffusers import DiffusionPipeline, AutoencoderTiny, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils import numpy_to_pil
import subprocess

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, safety_checker=None, vae=vae, variant="fp16", use_safetensors=True).to("cuda")
model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config,
                                                                       use_karras_sigmas=True)

for x in range(30):
    image = model("hurricane on earth", num_inference_steps=50).images[0]
    image.save("./" + str(x) + ".jpg", format='JPEG', quality=90)
