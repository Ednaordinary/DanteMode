import numpy as np
from diffusers.utils import numpy_to_pil

from .generic import GenericModel, GenericOutput, RunStatus, FinalOutput
from .optimized import OptimizedModel
from diffusers import AutoencoderTiny
import threading
import torch
import time
import gc


class IntermediateOutput(GenericOutput):
    pass


class IntermediateModel(GenericModel):
    def __init__(self, path, out_type, max_latent, steps, mini_vae):
        super().__init__(path, out_type, max_latent, steps)
        self.mini_vae = mini_vae
        self.mini_vae_path = mini_vae

    def to(self, device):
        super().to(device)
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)

    def del_model(self):
        del self.model
        del self.intermediates
        del self.mini_vae
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None
        self.mini_vae = self.mini_vae_path

    async def call(self, prompts):
        self.to("cuda")

        def intermediate_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            self.step = i
            self.intermediates = latents
            self.intermediate_update = True
            return kwargs

        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.out = self.model(prompts, negative_prompt=[x if x != None else "" for x in negative_prompts],
                                      num_inference_steps=steps, callback_on_step_end=callback,
                                      callback_on_step_end_tensor_inputs=[
                                          "latents"])  # callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
            except Exception as e:
                print(repr(e))
                self.out = [[]]
                pass

        for im in range(0, len(prompts), self.max_latent):
            current_prompts = prompts[im:im + self.max_latent]
            model_thread = threading.Thread(target=threaded_model,
                                            args=[[x.prompt for x in prompts[im:im + self.max_latent]],
                                                  [x.negative_prompt for x in prompts[im:im + self.max_latent]],
                                                  self.steps, intermediate_callback])
            model_thread.start()
            self.intermediates = None
            self.intermediate_update = False
            while model_thread.is_alive():
                if self.intermediate_update:
                    for idx, intermediate in enumerate(self.intermediates):
                        yield IntermediateOutput(output=intermediate, out_type="latent-image",
                                                 prompt=current_prompts[idx])
                    yield RunStatus(current=self.step,
                                    total=self.steps,
                                    interactions=[x.interaction for x in prompts[im:im + self.max_latent]])
                    self.intermediate_update = False
                time.sleep(0.01)
            outputs = []
            for idx, out in enumerate(self.out[0]):
                outputs.append(GenericOutput(output=out, out_type=self.out_type,
                                             prompt=current_prompts[idx]))
            yield FinalOutput(outputs=outputs)
            self.intermediates = None
            self.step = 0
            gc.collect()
            torch.cuda.empty_cache()


class IntermediateOptimizedModel(OptimizedModel):
    def __init__(self, path, out_type, max_latent, steps, mini_vae):
        super().__init__(path, out_type, max_latent, steps)
        self.mini_vae = mini_vae
        self.mini_vae_path = mini_vae

    def to(self, device):
        super().to(device)
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)
        #self.mini_vae.enable_slicing() # bugged? or something
        #self.mini_vae.enable_tiling()

    def del_model(self):
        del self.model
        del self.intermediates
        del self.mini_vae
        del self.helper
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None
        self.mini_vae = self.mini_vae_path

    async def call(self, prompts):
        self.to("cuda")
        self.helper.set_params(cache_interval=2, cache_branch_id=0)
        self.helper.enable()

        #self.stack = []

        def intermediate_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            self.step = i
            self.intermediates = latents
            self.intermediate_update = True
            return kwargs

        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.out = self.model(prompts, negative_prompt=[x if x != None else "" for x in negative_prompts],
                                      num_inference_steps=steps, callback_on_step_end=callback,
                                      callback_on_step_end_tensor_inputs=[
                                          "latents"])  # callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
            except Exception as e:
                print(repr(e))
                self.out = [[]]
                pass

        for im in range(0, len(prompts), self.max_latent):
            current_prompts = prompts[im:im + self.max_latent]
            model_thread = threading.Thread(target=threaded_model,
                                            args=[[x.prompt for x in prompts[im:im + self.max_latent]],
                                                  [x.negative_prompt for x in prompts[im:im + self.max_latent]],
                                                  self.steps, intermediate_callback])
            model_thread.start()
            self.intermediates = None
            self.intermediate_update = False
            while model_thread.is_alive():
                if self.intermediate_update:
                    for idx, intermediate in enumerate(self.intermediates):
                        yield IntermediateOutput(output=intermediate, out_type="latent-image",
                                                 prompt=current_prompts[idx])
                    yield RunStatus(current=self.step,
                                    total=self.steps,
                                    interactions=[x.interaction for x in prompts[im:im + self.max_latent]])
                    self.intermediate_update = False
                time.sleep(0.01)
            outputs = []
            for idx, out in enumerate(self.out[0]):
                outputs.append(GenericOutput(output=out, out_type=self.out_type,
                                             prompt=current_prompts[idx]))
            yield FinalOutput(outputs=outputs)
            #Image.fromarray(np.vstack(self.stack)).show()
            self.intermediates = None
            self.step = 0
            gc.collect()
            torch.cuda.empty_cache()
