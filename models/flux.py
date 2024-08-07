import gc
import json
import os
import sys
import threading
import time
from typing import Union, List, Optional, Dict, Callable, Any

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel, FluxPipeline
from diffusers.loaders import SD3LoraLoaderMixin
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast, T5EncoderModel
from optimum.quanto import freeze, quantize, qint8
import torch

from models.generic import GenericModel, GenericOutput, FinalOutput, RunStatus

class QKVFusedFluxTransformer2DModel(FluxTransformer2DModel):
    #copied and lightly modified from SD3 transformer code
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors
    def fuse_qkv_projections(self):
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors().items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)
    def unfuse_qkv_projections(self):
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

class FLUXDevModel(GenericModel):
    def __init__(self, path, out_type, max_latent, steps, guidance_scale, local_path):
        super().__init__(path, out_type, max_latent, steps)
        self.local_path = local_path
        self.guidance_scale = guidance_scale

    def to(self, device):
        dtype = torch.bfloat16
        #dtype = torch.float8_e4m3fn
        try:
            if not self.model:
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.path, subfolder="scheduler")
                text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
                text_encoder_2 = T5EncoderModel.from_pretrained(self.path, subfolder="text_encoder_2",
                                                                torch_dtype=dtype)
                #text_encoder_2.to(torch.float8_e4m3fn)
                tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype)
                vae = AutoencoderKL.from_pretrained(self.local_path, subfolder="vae", torch_dtype=dtype, device=device)
                transformer = FluxTransformer2DModel.from_pretrained(self.local_path, subfolder="transformer", torch_dtype=torch.float8_e4m3fn)
                self.model = FluxPipeline(
                    scheduler=scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=transformer,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                )
        except:
            gc.collect()
            torch.cuda.empty_cache()
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.path, subfolder="scheduler")
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            text_encoder_2 = T5EncoderModel.from_pretrained(self.path, subfolder="text_encoder_2",
                                                            torch_dtype=dtype)
            #text_encoder_2.to(torch.float8_e4m3fn)
            tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype)
            vae = AutoencoderKL.from_pretrained(self.local_path, subfolder="vae", torch_dtype=dtype, device=device)
            transformer = FluxTransformer2DModel.from_pretrained(self.local_path, subfolder="transformer",
                                                                 torch_dtype=torch.float8_e4m3fn)
            self.model = FluxPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                vae=vae,
            )
        self.model = self.model.to(device)
        #self.model.enable_model_cpu_offload()
        self.model.vae.enable_slicing()

    async def call(self, prompts):
        self.to("cuda")

        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                #Flux in diffusers doesnt support negative_prompt rn :(
                gc.collect()
                torch.cuda.empty_cache()
                self.out = self.model(prompts, num_inference_steps=steps,
                                      guidance_scale=self.guidance_scale,
                                      callback_on_step_end=callback,
                                      callback_on_step_end_tensor_inputs=[
                                          "latents"])
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(repr(e))
                self.out = [[]]
                #pass
                raise e

        def progress_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            self.step = i
            return kwargs

        for i in range(0, len(prompts), self.max_latent):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[[x.prompt for x in prompts[i:i + self.max_latent]],
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

class FLUXDevTempModel(GenericModel):
    def __init__(self, path, out_type, max_latent, steps, transformer, text_encoder_2, guidance_scale, max_seq):
        super().__init__(path, out_type, max_latent, steps)
        self.transformer = transformer
        self.text_encoder_2 = text_encoder_2
        self.guidance_scale = guidance_scale
        self.max_seq = max_seq

    def to(self, device):
        dtype = torch.bfloat16
        try:
            if not self.model:
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.path, subfolder="scheduler")
                text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
                #text_encoder_2 = QuantizedT5EncoderModel.from_pretrained(model_name_or_path="models/flux-d/text_encoder_2")
                #text_encoder_2 = T5EncoderModel.from_pretrained("models/flux-d/text_encoder_2")
                tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype)
                vae = AutoencoderKL.from_pretrained(self.path, subfolder="vae", torch_dtype=dtype, device=device)
                self.model = FluxPipeline(
                    scheduler=scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=self.transformer,
                    text_encoder_2=self.text_encoder_2,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                )
        except Exception as e:
            print(repr(e))
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.path, subfolder="scheduler")
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            # text_encoder_2 = QuantizedT5EncoderModel.from_pretrained(model_name_or_path="models/flux-d/text_encoder_2")
            # text_encoder_2 = T5EncoderModel.from_pretrained("models/flux-d/text_encoder_2")
            tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype)
            vae = AutoencoderKL.from_pretrained(self.path, subfolder="vae", torch_dtype=dtype, device=device)
            self.model = FluxPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=self.transformer,
                text_encoder_2=self.text_encoder_2,
                tokenizer_2=tokenizer_2,
                vae=vae,
            )
        # #self.model = self.model.to(device)
        # if device == "cuda":
        #     pass
        # else:
        self.model.to(device)
        #self.model.enable_model_cpu_offload()
        self.model.vae.enable_slicing()
        self.model.vae.enable_tiling()

    def FluxFix(self):
        self.transformer.to("cpu")
        self.text_encoder_2.to("cpu")

    def del_model(self):
        self.FluxFix()
        super().del_model()

    async def call(self, prompts):
        self.to("cuda")

        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                #Flux in diffusers doesnt support negative_prompt rn :(
                gc.collect()
                torch.cuda.empty_cache()
                self.out = self.model(prompts, num_inference_steps=steps,
                                      callback_on_step_end=callback,
                                      callback_on_step_end_tensor_inputs=[
                                          "latents"], guidance_scale=self.guidance_scale, max_sequence_length=self.max_seq)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(repr(e))
                self.out = [[]]
                #pass
                raise e

        def progress_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            self.step = i
            return kwargs

        for i in range(0, len(prompts), self.max_latent):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[[x.prompt for x in prompts[i:i + self.max_latent]],
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
