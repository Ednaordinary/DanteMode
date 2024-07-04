import threading
import time

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

from .generic import GenericOutput, FinalOutput, GenericModel, RunStatus


class SAUDIOModel(GenericModel):
    def to(self, device):
        try:
            self.model
        except:
            self.model, self.config = get_pretrained_model(self.path)
        else:
            if not self.model:
                self.model, self.config = get_pretrained_model(self.path)
        self.model = self.model.to(device)
        self.sample_rate = self.config["sample_rate"]
        self.sample_size = self.config["sample_size"]

    async def call(self, prompts):
        self.to("cuda")

        def threaded_model(model, prompts, negative_prompts, callback):
            try:
                #self.out = model(prompts, negative_prompt=negative_prompts, num_inference_steps=steps,
                #                 callback=callback, callback_steps=1)
                conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": 45} for prompt in prompts]
                self.out = generate_diffusion_cond(model, steps=self.steps, cfg_scale=7, conditioning=conditioning,
                                                   sample_size=self.sample_size, sigma_min=0.3, sigma_max=500,
                                                   sampler_type="dpmpp-3m-sde", device="cuda", callback=callback,
                                                   batch_size=len(conditioning))
                out = []
                for output in self.out:
                    output = rearrange(output.unsqueeze(0), "b d n -> d (b n)")
                    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(
                        torch.int16).cpu()
                    out.append(output)
                print(len(out))
                self.out = out
            except Exception as e:
                print(repr(e))
                self.out = [[]]
                pass

        def progress_callback(*args, **kwargs):
            self.step = args[0]['i']

        for i in range(0, len(prompts), self.max_latent):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[self.model, [x.prompt for x in prompts[i:i + self.max_latent]],
                                                  [x.negative_prompt for x in prompts[i:i + self.max_latent]],
                                                  progress_callback])
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
            for idx, out in enumerate(self.out):
                outputs.append(
                    GenericOutput(output=out, out_type=self.out_type, prompt=prompts[i:i + self.max_latent][idx]))
            yield FinalOutput(outputs=outputs)
