import threading
import time

from DeepCache import DeepCacheSDHelper

from .generic import GenericOutput, FinalOutput, RunStatus, GenericModel
from .intermediate import IntermediateOptimizedModel, IntermediateModel, IntermediateOutput
from diffusers import AutoencoderKL, AutoencoderTiny, DiffusionPipeline, DPMSolverMultistepScheduler, \
    AutoPipelineForText2Image, StableDiffusion3Pipeline
import torch
import gc

class SDXLModel(IntermediateOptimizedModel):
    def to(self, device):
        if not self.model:
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None, vae=vae, variant="fp16", use_safetensors=True)
            del vae
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config,
                                                                       use_karras_sigmas=True)  # , use_lu_lambdas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.helper = DeepCacheSDHelper(pipe=self.model)
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)

class SDXLTModel(GenericModel):
    def to(self, device):
        self.model = AutoPipelineForText2Image.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None, variant="fp16", use_safetensors=True)
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()
    async def call(self, prompts):
        self.to("cuda")
        for i in range(0, len(prompts), self.max_latent):
            #For SDXL Turbo we don't bother with run status's, it's too fast and just rate limits us
            try:
                self.out = self.model([x.prompt for x in prompts[i:i + self.max_latent]], negative_prompt=[x.negative_prompt for x in prompts[i:i + self.max_latent]], num_inference_steps=self.steps, guidance_scale=0.0)
            except:
                self.out = [[]]
            outputs = []
            for idx, out in enumerate(self.out[0]):
                outputs.append(GenericOutput(output=out, out_type=self.out_type, prompt=prompts[i:i + self.max_latent][idx]))
            yield FinalOutput(outputs=outputs)

class SD3Model(IntermediateModel):
    def to(self, device):
        if not self.model:
            self.model = StableDiffusion3Pipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None, use_safetensors=True)
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)