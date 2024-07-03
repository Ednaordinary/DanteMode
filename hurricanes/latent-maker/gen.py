import torch
from diffusers import DiffusionPipeline, AutoencoderTiny, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils import numpy_to_pil
import subprocess

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
mini_vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to("cuda")

model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, safety_checker=None, vae=vae, variant="fp16", use_safetensors=True).to("cuda")
model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config,
                                                                       use_karras_sigmas=True)

for x in range(10):
    def intermediate_callback(pipe, i, t, kwargs):
                latents = kwargs["latents"]
                decoded = mini_vae.decode(latents).sample[0]
                tmp_image = decoded.to('cpu', non_blocking=False)
                tmp_image = numpy_to_pil((tmp_image / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy())[0]
                tmp_image.save("./" + str(x) + "/" + str(i) + ".jpg", format='JPEG', quality=90)
                return kwargs
    model("hurricane on earth in the ocean", callback_on_step_end=intermediate_callback, callback_on_step_end_tensor_inputs=["latents"], num_inference_steps=50)
    subprocess.call("ffmpeg -i -y ./" + str(x) + "/%d.jpg ./" + str(x) + "/hurricane.mp4", shell=True)
