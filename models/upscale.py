import threading
import time

from PIL import Image
from diffusers import LDMSuperResolutionPipeline, DPMSolverMultistepScheduler
import torch
from models.generic import GenericModel, RunStatus, GenericOutput, FinalOutput


class LDMUpscaleModel(GenericModel):
    def to(self, device):
        try:
            self.model
        except:
            self.model = LDMSuperResolutionPipeline.from_pretrained(self.path, torch_dtype=torch.float16)
        else:
            if not self.model:
                self.model = LDMSuperResolutionPipeline.from_pretrained(self.path, torch_dtype=torch.float16)
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config,
                                                                       use_karras_sigmas=True)  # , use_lu_lambdas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.model = self.model.to(device)

    async def call(self, prompts):
        self.to("cuda")
        for i in range(0, len(prompts), self.max_latent):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[self, self.model, [x.prompt for x in prompts[i:i + self.max_latent]]])
            model_thread.start()
            step = 0
            self.step = 0
            try:
                self.out = self.model(prompts, num_inference_steps=self.steps)
            except:
                self.out = [[]]
            outputs = []
            for idx, out in enumerate(self.out[0]):
                outputs.append(
                    GenericOutput(output=out, out_type=self.out_type, prompt=prompts[i:i + self.max_latent][idx]))
            yield FinalOutput(outputs=outputs)