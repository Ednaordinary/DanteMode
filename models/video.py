import gc
import threading
import time

import torch
from DeepCache import DeepCacheSDHelper
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableVideoDiffusionPipeline, AutoencoderKL

from .generic import RunStatus, GenericOutput, FinalOutput
from .optimized import OptimizedModel


class ZSVideoModel(OptimizedModel):

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
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config,
                                                                       use_karras_sigmas=True)  #, use_lu_lambdas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.helper = DeepCacheSDHelper(pipe=self.model)

    def del_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")
        self.helper.set_params(cache_interval=2, cache_branch_id=0)
        self.helper.enable()

        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.out = self.model(prompts, negative_prompt=negative_prompts, num_inference_steps=steps,
                                 callback=callback, callback_steps=1, height=320, width=576, num_frames=24)
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

class SVDVideoModel(OptimizedModel):

    def to(self, device):
        try:
            self.model
        except:
            self.model = StableVideoDiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None)
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            self.image_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, safety_checker=None,
                                                           vae=vae, variant="fp16", use_safetensors=True)
        else:
            if not self.model:
                self.model = StableVideoDiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                               safety_checker=None)
                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
                self.image_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                     torch_dtype=torch.float16, safety_checker=None,
                                                                     vae=vae, variant="fp16", use_safetensors=True)
        self.model = self.model.to(device)
        self.image_model = self.image_model.to(device)
        self.image_model.vae.enable_slicing()
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config,
                                                                       use_karras_sigmas=True)
        self.image_model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config,
                                                                       use_karras_sigmas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.image_model.scheduler.algorithm_type = "dpmsolver++"
        self.helper = DeepCacheSDHelper(pipe=self.model)
        self.helper2 = DeepCacheSDHelper(pipe=self.image_model)

    def del_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")
        self.helper.set_params(cache_interval=2, cache_branch_id=0)
        self.helper.enable()
        self.helper2.set_params(cache_interval=1, cache_branch_id=0)
        self.helper2.enable()
        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.out = self.image_model
                self.out = self.model(prompts, negative_prompt=negative_prompts, num_inference_steps=steps,
                                 callback=callback, callback_steps=1, height=320, width=576, num_frames=24)
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