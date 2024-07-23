import gc
import os
import sys
import threading
import time

import torch
from DeepCache import DeepCacheSDHelper
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableVideoDiffusionPipeline, AutoencoderKL, \
    AutoencoderTiny

from .generic import RunStatus, GenericOutput, FinalOutput
from .intermediate import IntermediateOutput, IntermediateOptimizedModel, IntermediateModel
from .optimized import OptimizedModel
from rembg import remove

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
        del self.helper
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")
        self.helper.set_params(cache_interval=3, cache_branch_id=0)
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


class SVDVideoModel(IntermediateModel):

    def to(self, device):
        try:
            self.model
        except:
            self.model = StableVideoDiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                                      safety_checker=None)
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            self.image_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                 torch_dtype=torch.float16, safety_checker=None,
                                                                 vae=vae, variant="fp16", use_safetensors=True)
        else:
            if not self.model:
                self.model = StableVideoDiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                                          safety_checker=None)
                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
                self.image_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                     torch_dtype=torch.float16, safety_checker=None,
                                                                     vae=vae, variant="fp16", use_safetensors=True)
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)
        self.model = self.model.to("cpu")
        self.image_model.scheduler = DPMSolverMultistepScheduler.from_config(self.image_model.scheduler.config,
                                                                             use_karras_sigmas=True)
        self.image_model.scheduler.algorithm_type = "dpmsolver++"
        self.image_model = self.image_model.to(device)
        self.image_model.vae.enable_slicing()

    def del_model(self):
        del self.model
        del self.image_model
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")

        def image_threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.model.to("cpu")
                self.image_model.to("cuda")
                for x in self.image_model(prompts, negative_prompt=[x if x != None else "" for x in negative_prompts],
                                          num_inference_steps=steps, callback_on_step_end=callback,
                                          callback_on_step_end_tensor_inputs=[
                                              "latents"], height=576, width=1024).images:
                    self.out.append(x)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(repr(e))
                self.out = [[]]
                pass

        def threaded_model(prompts, steps, callback):
            try:
                self.image_model.to("cpu")
                self.model.to("cuda")
                self.out = self.model(prompts, num_inference_steps=steps, callback_on_step_end=callback,
                                      callback_on_step_end_tensor_inputs=["latents"], fps=7, num_frames=25,
                                      decode_chunk_size=10, min_guidance_scale=0.0, max_guidance_scale=0.0,
                                      motion_bucket_id=127).frames
            except Exception as e:
                print(repr(e))
                self.out = [[]]
                pass

        def progress_callback(pipe, i, t, kwargs):
            self.step = i
            return kwargs

        def intermediate_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            self.image_step = i
            self.intermediates = latents
            self.intermediate_update = True
            return kwargs

        for i in range(0, len(prompts), 10):
            self.out = []
            image_model_thread = threading.Thread(target=image_threaded_model,
                                                  args=[[x.prompt for x in prompts[i:i + 10]],
                                                        [x.negative_prompt for x in prompts[i:i + 10]],
                                                        self.steps, intermediate_callback])
            image_model_thread.start()
            self.image_step = 0
            self.intermediates = None
            self.intermediate_update = False
            while image_model_thread.is_alive():
                if self.intermediate_update:
                    for idx, intermediate in enumerate(self.intermediates):
                        yield IntermediateOutput(output=intermediate, out_type="latent-image",
                                                 prompt=prompts[i:i + 10][idx])
                    yield RunStatus(current=self.image_step / 2,
                                    total=self.steps,
                                    interactions=[x.interaction for x in prompts[i:i + 10]])
                    self.intermediate_update = False
                time.sleep(0.01)
            for i, out in enumerate(self.out):
                yield IntermediateOutput(output=out, out_type="image",
                                         prompt=prompts[i:i + 10][i])
            yield RunStatus(current=self.steps / 2,
                            total=self.steps,
                            interactions=[x.interaction for x in prompts[i:i + 10]])
        for i, out in enumerate(self.out):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[out, self.steps, progress_callback])
            model_thread.start()
            step = 0
            self.step = 0
            while model_thread.is_alive():
                if step != self.step:
                    yield RunStatus(current=(self.step + self.steps) / 2,
                                    total=self.steps,
                                    interactions=[prompts[i].interaction])
                    step = self.step
                time.sleep(0.01)
            outputs = [GenericOutput(output=self.out[0], out_type=self.out_type, prompt=prompts[i])]
            yield FinalOutput(outputs=outputs)

class SV3DVideoModel(IntermediateModel):

    def to(self, device):
        try:
            self.model
        except:
            self.model = StableVideoDiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                                      safety_checker=None)
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            self.image_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                 torch_dtype=torch.float16, safety_checker=None,
                                                                 vae=vae, variant="fp16", use_safetensors=True)
        else:
            if not self.model:
                self.model = StableVideoDiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                                          safety_checker=None)
                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
                self.image_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                                     torch_dtype=torch.float16, safety_checker=None,
                                                                     vae=vae, variant="fp16", use_safetensors=True)
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)
        self.model = self.model.to("cpu")
        self.image_model.scheduler = DPMSolverMultistepScheduler.from_config(self.image_model.scheduler.config,
                                                                             use_karras_sigmas=True)
        self.image_model.scheduler.algorithm_type = "dpmsolver++"
        self.image_model = self.image_model.to(device)
        self.image_model.vae.enable_slicing()

    def del_model(self):
        del self.model
        del self.image_model
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")

        def image_threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.model.to("cpu")
                self.image_model.to("cuda")
                for x in self.image_model(prompts, negative_prompt=[x if x != None else "" for x in negative_prompts],
                                          num_inference_steps=steps, callback_on_step_end=callback,
                                          callback_on_step_end_tensor_inputs=[
                                              "latents"], height=576, width=576).images:
                    self.out.append(x)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(repr(e))
                self.out = [[]]
                pass

        def threaded_model(prompts, steps, callback):
            try:
                self.image_model.to("cpu")
                self.model.to("cuda")
                self.out = self.model(prompts, num_inference_steps=steps, callback_on_step_end=callback,
                                      callback_on_step_end_tensor_inputs=["latents"], fps=7, num_frames=21,
                                      decode_chunk_size=10, min_guidance_scale=0.0, max_guidance_scale=0.0).frames
            except Exception as e:
                print(repr(e))
                self.out = [[]]
                pass

        def progress_callback(pipe, i, t, kwargs):
            self.step = i
            return kwargs

        def intermediate_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            self.image_step = i
            self.intermediates = latents
            self.intermediate_update = True
            return kwargs

        for i in range(0, len(prompts), 10):
            self.out = []
            image_model_thread = threading.Thread(target=image_threaded_model,
                                                  args=[[x.prompt for x in prompts[i:i + 10]],
                                                        [x.negative_prompt for x in prompts[i:i + 10]],
                                                        self.steps, intermediate_callback])
            image_model_thread.start()
            self.image_step = 0
            self.intermediates = None
            self.intermediate_update = False
            while image_model_thread.is_alive():
                if self.intermediate_update:
                    for idx, intermediate in enumerate(self.intermediates):
                        yield IntermediateOutput(output=intermediate, out_type="latent-image",
                                                 prompt=prompts[i:i + 10][idx])
                    yield RunStatus(current=self.image_step / 2,
                                    total=self.steps,
                                    interactions=[x.interaction for x in prompts[i:i + 10]])
                    self.intermediate_update = False
                time.sleep(0.01)
            for i, out in enumerate(self.out):
                yield IntermediateOutput(output=out, out_type="image",
                                         prompt=prompts[i:i + 10][i])
            yield RunStatus(current=self.steps / 2,
                            total=self.steps,
                            interactions=[x.interaction for x in prompts[i:i + 10]])
        for i in range(0, len(prompts), 10):
            out = []
            for idx, intermediate in enumerate([remove(x) for x in self.out]):
                yield IntermediateOutput(output=intermediate, out_type="latent-image",
                                         prompt=prompts[i:i + 10][idx])
                out.append(intermediate)
            self.out = out
        for i, out in enumerate(self.out):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[out, self.steps, progress_callback])
            model_thread.start()
            step = 0
            self.step = 0
            while model_thread.is_alive():
                if step != self.step:
                    yield RunStatus(current=(self.step + self.steps) / 2,
                                    total=self.steps,
                                    interactions=[prompts[i].interaction])
                    step = self.step
                time.sleep(0.01)
            outputs = [GenericOutput(output=self.out[0], out_type=self.out_type, prompt=prompts[i])]
            yield FinalOutput(outputs=outputs)
