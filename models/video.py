import gc
import threading
import time

import torch
from DeepCache import DeepCacheSDHelper
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from .generic import RunStatus, GenericOutput, FinalOutput
from .optimized import OptimizedModel


class ZSVideoModel(OptimizedModel):

    def to(self, device):
        try:
            self.model
        except:
            self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None)
            self.upscale = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", torch_dtype=torch.float16)
        else:
            if not self.model:
                self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                               safety_checker=None)
                self.upscale = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", torch_dtype=torch.float16)
        self.model = self.model.to(device)
        self.upscale = self.upscale.to(device)
        self.model.vae.enable_slicing()
        self.upscale.vae.enable_slicing()
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config,
                                                                       use_karras_sigmas=True)  #, use_lu_lambdas=True)
        self.upscale.scheduler = DPMSolverMultistepScheduler.from_config(self.upscale.scheduler.config,
                                                                       use_karras_sigmas=True)  #, use_lu_lambdas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.helper = DeepCacheSDHelper(pipe=self.model)
        self.helper2 = DeepCacheSDHelper(pipe=self.upscale)

    def del_model(self):
        del self.model
        del self.upscale
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")

        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.model.to("cuda")
                self.out = self.model(prompts, negative_prompt=negative_prompts, num_inference_steps=steps,
                                 callback=callback, callback_steps=1, height=320, width=576, num_frames=24, output_type="latent")
                print(self.out.frames.shape)
                self.model.to("cpu")
                #self.out = []
                #self.out = [Image.fromarray(frame).resize((1024, 576)) for frame in self.out]
                self.out = self.upscale(prompts, video=self.out.frames, strength=0.6).frames
            except Exception as e:
                print(repr(e))
                self.out = [[]]
                pass

        def progress_callback(i, t, latents):
            self.step = i

        for i in range(0, len(prompts), self.max_latent):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[[x.prompt for x in prompts[i:i + self.max_latent]],
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