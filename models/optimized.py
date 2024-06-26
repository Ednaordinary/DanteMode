from .generic import GenericModel
from diffusers import DPMSolverMultistepScheduler
from DeepCache import DeepCacheSDHelper

class OptimizedModel(GenericModel):
    def __init__(self, path, out_type, max_latent, steps):
        super().__init__(path, out_type, max_latent, steps)
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config, use_karras_sigmas=True) #, use_lu_lambdas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.model.vae.enable_slicing()
        self.helper = DeepCacheSDHelper(pipe=self.model)
    async def call(self, prompts):
        if self.model.device.type != "cuda": self.model.to("cuda")
        self.helper.set_params(cache_interval=3, cache_branch_id=0)
        self.helper.enable()
        super().call(prompts)
