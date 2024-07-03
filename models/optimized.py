import threading
import time

from .generic import GenericModel, RunStatus, GenericOutput, FinalOutput
from diffusers import DPMSolverMultistepScheduler
from DeepCache import DeepCacheSDHelper


class OptimizedModel(GenericModel):
    def to(self, device):
        super().to(device)
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config,
                                                                       use_karras_sigmas=True)  #, use_lu_lambdas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.helper = DeepCacheSDHelper(pipe=self.model)

    async def call(self, prompts):
        self.to("cuda")
        self.helper.set_params(cache_interval=2, cache_branch_id=0)
        self.helper.enable()

        def threaded_model(model, prompts, negative_prompts, steps, callback):
            try:
                self.out = model(prompts, negative_prompt=negative_prompts, num_inference_steps=steps,
                                 callback=callback, callback_steps=1)
            except Exception as e:
                print(repr(e))
                self.out = [[]]
                pass

        def progress_callback(i, t, latents):
            self.step = i

        for i in range(0, len(prompts), self.max_latent):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[self.model,
                                                  [x.prompt for x in prompts[i:i + self.max_latent]],
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
