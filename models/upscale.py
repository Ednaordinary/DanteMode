import os
import sys
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
            step = 0
            self.step = 0
            try:
                self.out = [self.model(prompt, num_inference_steps=self.steps, output_type="pil").images for prompt in prompts]
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(repr(e))
                self.out = [[]]
                pass
            outputs = []
            print(self.out)
            for idx, out in enumerate(self.out):
                outputs.append(
                    GenericOutput(output=out, out_type=self.out_type, prompt=prompts[i:i + self.max_latent][idx]))
            yield FinalOutput(outputs=outputs)