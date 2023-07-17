import sys
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None, feature_extractor=None)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
prompt = sys.argv[1]
images = pipe(prompt, num_images_per_prompt=int(sys.argv[2])).images
num = 0
for i in images:
    i.save(str(num) + ".jpg")
    num += 1
