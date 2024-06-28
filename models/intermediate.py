from .generic import GenericOutput, RunStatus, FinalOutput
from .optimized import OptimizedModel
from diffusers import AutoencoderTiny
import threading
import torch
import time
import gc


class IntermediateOutput(GenericOutput):
    pass


class IntermediateOptimizedModel(OptimizedModel):
    def __init__(self, path, out_type, max_latent, steps, mini_vae):
        super().__init__(path, out_type, max_latent, steps)
        self.mini_vae = mini_vae

    def to(self, device):
        super().to(device)
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)
        self.mini_vae.enable_slicing()
        self.mini_vae.enable_tiling()

    def del_model(self):
        del self.model
        del self.intermediates
        del self.mini_vae
        del self.helper
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")
        self.helper.set_params(cache_interval=3, cache_branch_id=0)
        self.helper.enable()

        def intermediate_callback(i, t, latents):
            #latents = kwargs["latents"]
            #print(latents)
            #sample = self.mini_vae.decode(latents).sample
            self.step = i
            self.intermediates = latents
            self.intermediate_update = True

        def threaded_model(prompts, negative_prompts, steps, callback):
            self.out = self.model(prompts, negative_prompt=[x if x != None else "" for x in negative_prompts],
                                  num_inference_steps=steps, callback=callback,
                                  callback_steps=1)  # callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])

        for i in range(0, len(prompts), self.max_latent):
            #output = self.model([x.prompt for x in prompts[i:i+self.max_latent]], negative_prompt=[x.negative_prompt for x in prompts[i:i+self.max_latent]], num_inference_steps=self.steps)
            model_thread = threading.Thread(target=threaded_model,
                                            args=[[x.prompt for x in prompts[i:i + self.max_latent]],
                                                  [x.negative_prompt for x in prompts[i:i + self.max_latent]],
                                                  self.steps, intermediate_callback])
            model_thread.start()
            self.intermediates = None
            self.intermediate_update = False
            while model_thread.is_alive():
                if self.intermediate_update:
                    for idx, intermediate in enumerate(self.intermediates):
                        yield IntermediateOutput(output=intermediate, out_type="latent-image",
                                                 prompt=prompts[i:i + self.max_latent][idx])
                    yield RunStatus(current=self.step,
                                    total=self.steps,
                                    interactions=[x.interaction for x in prompts[i:i + self.max_latent]])
                    self.intermediate_update = False
                time.sleep(0.01)
            outputs = []
            for idx, out in enumerate(self.out[0]):
                outputs.append(GenericOutput(output=out, out_type=self.out_type,
                                    prompt=prompts[i:i + self.max_latent][idx]))
            yield FinalOutput(outputs=outputs)
            self.intermediates = None
            self.step = 0
            gc.collect()
            torch.cuda.empty_cache()