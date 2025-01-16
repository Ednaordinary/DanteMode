import torch
import threading
import gc
import time
from .generic import RunStatus, GenericOutput, FinalOutput, GenericModel
from diffusers import AllegroPipeline, AutoencoderKLAllegro, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler

class AllegroModel(GenericModel):
    def __init__(self, path, out_type, max_latent, steps, revision):
        super().__init__(path, out_type, max_latent, steps)
        self.revision = revision
    def to(self, device):
        try:
            self.model
        except:
            vae = AutoencoderKLAllegro.from_pretrained(self.path, subfolder="vae", torch_dtype=torch.float32)
            self.model = AllegroPipeline.from_pretrained(self.path, torch_dtype=torch.bfloat16, vae=vae, revision=self.revision)
        else:
            if not self.model:
                vae = AutoencoderKLAllegro.from_pretrained(self.path, subfolder="vae", torch_dtype=torch.float32)
                self.model = AllegroPipeline.from_pretrained(self.path, torch_dtype=torch.bfloat16, vae=vae, revision=self.revision)
        self.model.scheduler = UniPCMultistepScheduler.from_config(self.model.scheduler.config)
        self.model.enable_model_cpu_offload()
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.vae.enable_slicing()
        self.model.vae.enable_tiling()

    def del_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")
        def progress_callback(pipe, i, t, kwargs):
            self.step = i + 1
            return kwargs

        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.out = self.model(prompts, negative_prompt=negative_prompts, num_inference_steps=steps,
                                      callback_on_step_end=callback,
                                      callback_on_step_end_tensor_inputs=["latents"], num_frames=88).frames
            except Exception as e:
                print(repr(e))
                self.out = []
                pass

        for i, prompt in enumerate(prompts):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[prompt.prompt,
                                                  prompt.negative_prompt,
                                                  self.steps, progress_callback])
            model_thread.start()
            step = 0
            self.step = 0
            while model_thread.is_alive():
                if step != self.step:
                    yield RunStatus(current=self.step,
                                    total=self.steps,
                                    interactions=[prompt.interaction])
                    step = self.step
                time.sleep(0.01)
            outputs = []
            for idx, out in enumerate(self.out):
                outputs.append(
                    GenericOutput(output=out, out_type=self.out_type, prompt=prompt))
            yield FinalOutput(outputs=outputs)
