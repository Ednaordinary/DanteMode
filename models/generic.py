from diffusers import DiffusionPipeline
import torch
import time
import threading
import gc


class RunStatus:
    def __init__(self, current, total, interactions):
        self.current = current
        self.total = total,
        self.interactions = interactions


class Prompt:
    def __init__(self, prompt, negative_prompt, interaction, index, parent_amount):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.interaction = interaction
        self.index = index
        self.parent_amount = parent_amount


class GenericOutput:
    def __init__(self, output, out_type, prompt):  #, interaction, index):
        self.output = output
        self.out_type = out_type,
        self.prompt = prompt


#        self.interaction = interaction
#        self.index = index

class FinalOutput:
    def __init__(self, outputs):
        self.outputs = outputs


class GenericModel:
    def __init__(self, path, out_type, max_latent, steps):
        self.path = path
        self.model = None
        self.out_type = out_type
        self.max_latent = max_latent
        self.steps = steps

    def to(self, device):
        try:
            self.model
        except:
            self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None)
        else:
            if not self.model:
                self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                               safety_checker=None)
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()

    def del_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")

        def threaded_model(model, prompts, negative_prompts, steps, callback):
            try:
                self.out = model(prompts, negative_prompt=negative_prompts, num_inference_steps=steps,
                                 callback=callback, callback_steps=1)
            except:
                self.out = [[]]
                pass

        def progress_callback(i, t, latents):
            self.step = i

        for i in range(0, len(prompts), self.max_latent):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[self.model, [x.prompt for x in prompts[i:i + self.max_latent]],
                                                  [x.negative_prompt for x in prompts[i:i + self.max_latent]],
                                                  self.steps, progress_callback])
            model_thread.start()
            step = 0
            self.step = 0
            while model_thread.is_alive():
                if step != self.step:
                    yield RunStatus(current=self.step,
                                    total=self.steps,
                                    interactions=[x.interaction for x in prompts[i:i + self.max_latent]])
                    step = self.step
                time.sleep(0.01)
            outputs = []
            for idx, out in enumerate(self.out[0]):
                outputs.append(
                    GenericOutput(output=out, out_type=self.out_type, prompt=prompts[i:i + self.max_latent][idx]))
            yield FinalOutput(outputs=outputs)
