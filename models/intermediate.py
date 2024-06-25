for generic import GenericModel
from diffusers import DPMSolverMultistepScheduler
from DeepCache import DeepCacheSDHelper

class OptimizedModel(GenericModel):
    def __init__(self, path, out_type, max_latent, steps):
        self.path = path
        self.model = DiffusionPipeline(path, torch_dtype=torch.float16)
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config, use_karras_sigmas=True) #, use_lu_lambdas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.helper = DeepCacheSDHelper(pipe=self.model)
        self.out_type = out_type
        self.max_latent = max_latent
        self.steps = steps
    async def call(self, prompts):
        if model.device.type != "cuda": self.model.to("cuda")
        self.helper.set_params(cache_interval=3, cache_branch_id=0)
        self.helper.enable()
        for i in range(0, len(prompts), self.max_latent):
            output = self.model(prompts[i:i+self.max_latent], num_inference_steps=self.steps)
            for idx, out in enumerate(output):
                yield GenericOutput(output=out, out_type=self.out_type, interaction=prompts[i:i+self.max_latent][idx])
