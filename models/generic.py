from diffusers import DiffusionPipeline
import torch

class Prompt:
    def __init__(self, prompt, negative_prompt, interaction):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.interaction = interaction

class GenericOutput:
    def __init__(self, output, out_type, interaction):
        self.output = output
        self.out_type = out_type,
        self.interaction = interaction

class GenericModel:
    def __init__(self, path, out_type, max_latent, steps):
        self.path = path
        self.model = DiffusionPipeline(path, torch_dtype=torch.float16)
        self.out_type = out_type
        self.max_latent = max_latent
        self.steps = steps
    def to(self, device):
        self.model = self.model.to(device)
    async def call(self, prompts):
        if model.device.type != "cuda": self.model.to("cuda")
        for i in range(0, len(prompts), self.max_latent):
            output = self.model(prompts[i:i+self.max_latent], num_inference_steps=self.steps)
            for idx, out in enumerate(output):
                yield GenericOutput(output=out, out_type=self.out_type, interaction=prompts[i:i+self.max_latent][idx])
