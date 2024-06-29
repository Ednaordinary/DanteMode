from DeepCache import DeepCacheSDHelper
from .intermediate import IntermediateOptimizedModel
from diffusers import AutoencoderKL, AutoencoderTiny, DiffusionPipeline, DPMSolverMultistepScheduler
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
        #self.mini_vae.enable_slicing()
        #self.mini_vae.enable_tiling()