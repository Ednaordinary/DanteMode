import torch
import torch.nn as nn
import threading
import gc
import time
from .generic import RunStatus, GenericOutput, FinalOutput, GenericModel
from .FLAVR.FLAVR import FLAVRModel
from diffusers import MochiPipeline, MochiTransformer3DModel, AutoencoderKLMochi
from diffusers.models.transformers.transformer_mochi import MochiTransformerBlock
#from diffusers.models.hooks import ModelHook, add_hook_to_module, PyramidAttentionBroadcastHook, remove_hook_from_module
from diffusers.models.attention_processor import Attention
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
import math

def enable_layerwise_upcasting(model, upcast_dtype=None, original_dtype=None):
    upcast_dtype = upcast_dtype or torch.float32
    original_dtype = original_dtype or model.dtype

    def upcast_dtype_hook_fn(module, *args, **kwargs):
        module = module.to(upcast_dtype)

    def cast_to_original_dtype_hook_fn(module, *args, **kwargs):
        module = module.to(original_dtype)

    def fn_recursive_upcast(module):
        # Upcast entire module and exist recursion
        module.register_forward_pre_hook(upcast_dtype_hook_fn)
        module.register_forward_hook(cast_to_original_dtype_hook_fn)

        #has_children = list(module.children())
        #if not has_children:
        #    module.register_forward_pre_hook(upcast_dtype_hook_fn)
        #    module.register_forward_hook(cast_to_original_dtype_hook_fn)

        #for child in module.children():
        #    fn_recursive_upcast(child)

    for module in model.children():
        fn_recursive_upcast(module)

class MochiModel(GenericModel):
    def __init__(self, path, out_type, max_latent, steps, guidance, length, flavr_path, dynamic_cfg):
        super().__init__(path, out_type, max_latent, steps)
        self.guidance = guidance
        self.length = length
        self.flavr_path = flavr_path
        self.dynamic_cfg = dynamic_cfg
    def to(self, device):
        try:
            self.model
        except:
            vae = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.bfloat16)
            transformer = MochiTransformer3DModel.from_pretrained(self.path, subfolder="transformer", torch_dtype=torch.bfloat16)
            mean_attns = []
            for idx, i in enumerate(transformer.modules()):
                if isinstance(i, MochiTransformerBlock) and idx != 16 and idx != 2084: # 2084 is the last layer, and should likely be skipped.
                    mean_attns.append((idx, torch.mean(torch.ravel(i.attn1.norm_k.weight))))
            mean_attns.sort(key=lambda x: x[1])
            attn_high = [x[0] for x in mean_attns[:int(len(mean_attns) // (10 / 5))]]
            idxs = attn_high
            for idx, i in enumerate(transformer.modules()):
                if isinstance(i, MochiTransformerBlock) and idx in idxs:
                    i.to(torch.float8_e4m3fn)
                    enable_layerwise_upcasting(i, upcast_dtype=torch.bfloat16, original_dtype=torch.float8_e4m3fn)
            self.model = MochiPipeline.from_pretrained(self.path, torch_dtype=torch.bfloat16, transformer=transformer, vae=vae)
            apply_cache_on_pipe(self.model, residual_diff_threshold=0.06)
            self.flavr = FLAVRModel(self.flavr_path)
        else:
            if not self.model:
                vae = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.bfloat16)
                transformer = MochiTransformer3DModel.from_pretrained(self.path, subfolder="transformer", torch_dtype=torch.bfloat16)
                mean_attns = []
                for idx, i in enumerate(transformer.modules()):
                    if isinstance(i, MochiTransformerBlock) and idx != 16 and idx != 2084: # 2084 is the last layer, and should likely be skipped.
                        mean_attns.append((idx, torch.mean(torch.ravel(i.attn1.norm_k.weight))))
                mean_attns.sort(key=lambda x: x[1])
                attn_high = [x[0] for x in mean_attns[:int(len(mean_attns) // (10 / 5))]]
                idxs = attn_high
                for idx, i in enumerate(transformer.modules()):
                    if isinstance(i, MochiTransformerBlock) and idx in idxs:
                        i.to(torch.float8_e4m3fn)
                        enable_layerwise_upcasting(i, upcast_dtype=torch.bfloat16, original_dtype=torch.float8_e4m3fn)
                self.model = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16, transformer=transformer, vae=vae)
                apply_cache_on_pipe(self.model)
                self.flavr = FLAVRModel(self.flavr_path)
        self.model.enable_model_cpu_offload()
        self.model.enable_vae_slicing()
        self.model.enable_vae_tiling()
        if device != "cuda":
            self.model.to(device)

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
        #10, 4.5, 7.5
        guidance_scale_begin = 8.5 # Large effect on temporal motion ?
        guidance_scale_mid = 6.5 # Should stay at ~4.5
        mid_point = 0.3 #between 0 and 1. 0.3 is a good default.
        guidance_scale_end = 7.5 # Large effect on spatial clarity
        steps=self.steps

        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            step_index = step_index + 1 # liar
            self.step = step_index
            if self.dynamic_cfg:
                if step_index > (steps*mid_point):
                    pipe._guidance_scale = (((guidance_scale_end-guidance_scale_mid) / math.pow(steps - (steps*mid_point),2)) * math.pow(step_index - (steps*mid_point),2)) + guidance_scale_mid
                else:
                    pipe._guidance_scale = (guidance_scale_begin-guidance_scale_mid)*(math.pow(step_index-(steps*mid_point),2)/math.pow(steps*mid_point,2)) + guidance_scale_mid
            #print("Current guidance scale:", pipe._guidance_scale)
            return callback_kwargs

        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                self.out = self.model(prompts, negative_prompt=negative_prompts, num_inference_steps=steps,
                                      callback_on_step_end=callback,
                                      callback_on_step_end_tensor_inputs=["latents", "prompt_embeds", "negative_prompt_embeds"],
                                      guidance_scale=self.guidance, height=480, width=848, num_frames=self.length).frames
                #for i in self.flavr(self.out):
                #    if isinstance(i, tuple):
                #        if not i[0] % 4:
                #            self.step += 1
                #    else:
                #        self.out = i
                for i in self.flavr(self.out):
                    if isinstance(i, tuple):
                        #if not i[0] % (4):
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
