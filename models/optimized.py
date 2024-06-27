from .generic import GenericModel
from diffusers import DPMSolverMultistepScheduler
from DeepCache import DeepCacheSDHelper


class OptimizedModel(GenericModel):
    def to(self, device):
        super().to(device)
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config,
                                                                       use_karras_sigmas=True)  #, use_lu_lambdas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.model.vae.enable_slicing()
        self.helper = DeepCacheSDHelper(pipe=self.model)

    async def call(self, prompts):
        self.to("cuda")
        self.helper.set_params(cache_interval=1, cache_branch_id=0)
        self.helper.enable()
        super().call(prompts)
