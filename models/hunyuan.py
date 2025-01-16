import torch
import torch.nn as nn
import threading
import gc
import time
from .generic import RunStatus, GenericOutput, FinalOutput, GenericModel
from .FLAVR.FLAVR import FLAVRModel
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, BitsAndBytesConfig, FlowMatchEulerDiscreteScheduler
import math

prompt_template = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the contents, including objects, people, and anything else."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the contents."
        "4. Background environment, light, style, atmosphere, and qualities."
        "5. Camera angles, movements, and transitions used in the video."
        "6. Thematic and aesthetic concepts associated with the scene, i.e. realistic, futuristic, fairy tale, etc<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}

class HNModel(GenericModel):
    def __init__(self, path, transformerpath, out_type, max_latent, steps, guidance, length, flavr_path, shift):
        super().__init__(path, out_type, max_latent, steps)
        self.guidance = guidance
        self.length = length
        self.flavr_path = flavr_path
        self.transformerpath = transformerpath
        self.shift = shift
    def to(self, device):
        try:
            self.model
        except:
            self.model = HunyuanVideoPipeline.from_pretrained(self.path, transformer=None, revision="refs/pr/18", torch_dtype=torch.float16)
            self.flavr = FLAVRModel(self.flavr_path)
        else:
            if not self.model:
                self.model = HunyuanVideoPipeline.from_pretrained(self.path, transformer=None, revision="refs/pr/18", torch_dtype=torch.float16)
                self.flavr = FLAVRModel(self.flavr_path)
        self.model.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.model.scheduler.config, shift=self.shift)
        self.model.vae.enable_tiling()
        if device != "cuda":
            self.model.to(device)
        else:
            if self.model.transformer == None:
                self.model.transformer = HunyuanVideoTransformer3DModel.from_pretrained(self.transformerpath, torch_dtype=torch.bfloat16)
                self.model.enable_model_cpu_offload()

    def del_model(self):
        try:
            del self.model.transformer
        except:
            pass
        try:
            del self.model
        except:
            pass
        try:
            del self.flavr
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            step_index = step_index + 1 # liar
            self.step = step_index
            return callback_kwargs


        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.out = self.model(prompts, num_inference_steps=steps,
                                      callback_on_step_end=callback,
                                      callback_on_step_end_tensor_inputs=[],
                                      prompt_template=prompt_template,
                                      guidance_scale=self.guidance, height=544, width=960, num_frames=self.length).frames
                print("starting flavr run")
                for i in self.flavr(self.out):
                    if isinstance(i, tuple):
                        #if not i[0] % (8):
                        #    self.step += 1
                        pass
                    else:
                        self.out = [i]
            except Exception as e:
                print(repr(e))
                self.out = []
                raise

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
