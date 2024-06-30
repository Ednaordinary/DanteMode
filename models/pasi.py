import gc
import threading
import time

from diffusers import PixArtSigmaPipeline, Transformer2DModel, AutoencoderTiny

from .generic import RunStatus, FinalOutput, GenericOutput
from .intermediate import IntermediateModel, IntermediateOutput
import torch

class PASIModel(IntermediateModel):
    def to(self, device):
        if not self.model:
            self.transformer = Transformer2DModel.from_pretrained(
                "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                subfolder='transformer',
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            self.model = PixArtSigmaPipeline.from_pretrained(self.path, torch_dtype=torch.float16, transformer=self.transformer, use_safetensors=True)
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)

    async def call(self, prompts):
        self.to("cuda")
        #self.stack = []

        def intermediate_callback(i, t, latents):
            #pixart sigma doesnt support callback on step end
            self.step = i
            self.intermediates = latents
            self.intermediate_update = True

        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.out = self.model(prompts, negative_prompt=[x if x != None else "" for x in negative_prompts],
                                      num_inference_steps=steps, callback=intermediate_callback, callback_steps=1, height=4096, width=4096)
            except Exception as e:
                print(repr(e))
                self.out = [[]]
        for im in range(0, len(prompts), self.max_latent):
            #output = self.model([x.prompt for x in prompts[i:i+self.max_latent]], negative_prompt=[x.negative_prompt for x in prompts[i:i+self.max_latent]], num_inference_steps=self.steps)
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