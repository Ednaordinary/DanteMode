from diffusers import DiffusionPipeline
import torch
import threading

class RunStatus:
    def __init__(self, current, total):
        self.current = current
        self.total = total

class Prompt:
    def __init__(self, prompt, negative_prompt, interaction, index):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.interaction = interaction
        self.index = index

class GenericOutput:
    def __init__(self, output, out_type, interaction, index):
        self.output = output
        self.out_type = out_type,
        self.interaction = interaction
        self.index = index

class GenericModel:
    def __init__(self, path, out_type, max_latent, steps):
        self.path = path
        self.model = DiffusionPipeline(path, torch_dtype=torch.float16)
        self.model.vae.enable_slicing()
        self.out_type = out_type
        self.max_latent = max_latent
        self.steps = steps
    def to(self, device):
        self.model = self.model.to(device)
    async def call(self, prompts):
        if model.device.type != "cuda": self.model.to("cuda")
        def threaded_model(self, model, prompts, negative_prompts, steps):
            self.out = model(prompts, negative_prompt=negative_prompts, num_inference_steps=steps)
        for i in range(0, len(prompts), self.max_latent):
            model_thread = threading.Thread(target=threaded_model, args=[self, self.model, [x.prompt for x in prompts[i:i+self.max_latent]], [x.negative_prompt for x in prompts[i:i+self.max_latent]], self.steps])
            model_thread.start()
            while model_thread.is_alive():
                
            for idx, out in enumerate(output):
                yield GenericOutput(output=out, out_type=self.out_type, interaction=prompts[i:i+self.max_latent][idx].interaction, index=prompts[i:i+self.max_latent][idx].index)
