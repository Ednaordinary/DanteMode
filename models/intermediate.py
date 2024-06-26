from .generic import GenericOutput, RunStatus
from .optimized import OptimizedModel
from diffusers import AutoencoderTiny
import threading
import torch
import time
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
    async def call(self, prompts):
        self.to("cuda")
        self.helper.set_params(cache_interval=3, cache_branch_id=0)
        self.helper.enable()
        def intermediate_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            sample = self.mini_vae.decode(latents).sample
            self.step = i
            self.intermediates = sample
        def threaded_model(prompts, negative_prompts, steps, callback):
            print(prompts, negative_prompts, steps, callback)
            self.out = self.model(prompts, negative_prompt=[x if x != None else "" for x in negative_prompts], num_inference_steps=steps, callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
        for i in range(0, len(prompts), self.max_latent):
            #output = self.model([x.prompt for x in prompts[i:i+self.max_latent]], negative_prompt=[x.negative_prompt for x in prompts[i:i+self.max_latent]], num_inference_steps=self.steps)
            model_thread = threading.Thread(target=threaded_model, args=[[x.prompt for x in prompts[i:i+self.max_latent]], [x.negative_prompt for x in prompts[i:i+self.max_latent]], self.steps, intermediate_callback])
            model_thread.start()
            intermediates = None
            self.intermediates = None
            while model_thread.is_alive():
                if self.intermediates != intermediates:
                    for idx, intermediate in enumerate(self.intermediates):
                        #intermediate = intermediate.to('cpu', non_blocking=True)
                        #intermediate = numpy_to_pil((intermediate / 2 + 0.5).permute(1, 2, 0).numpy())[0].resize((256, 256))
                        #intermediates should be handled only when we actually want to send them
                        yield IntermediateOutput(output=intermediate, out_type="latent-image", interaction=prompts[i:i+self.max_latent][idx].interaction, index=prompts[i:i+self.max_latent][idx].index)
                    yield RunStatus(current=self.step, total=self.steps, interactions=[x.interaction for x in prompts[i:i+self.max_latent]])
                    intermediates = self.intermediates
                time.sleep(0.01)
            for idx, out in enumerate(self.out):
                yield GenericOutput(output=out, out_type=self.out_type, interaction=prompts[i:i+self.max_latent][idx].interaction, index=prompts[i:i+self.max_latent][idx].index)
