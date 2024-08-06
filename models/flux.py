import gc
import json
import os
import sys
import threading
import time
from typing import Union, List, Optional, Dict, Callable

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel, FluxPipeline
from diffusers.loaders import SD3LoraLoaderMixin
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast, T5EncoderModel
from optimum.quanto import freeze, quantize, qint8
import torch

from models.generic import GenericModel, GenericOutput, FinalOutput, RunStatus

#from accelerate import infer_auto_device_map

class FluxFp8Pipeline(FluxPipeline):
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=self.text_encoder.dtype)

        prompt_embeds = prompt_embeds.to(torch.float8_e4m3fn)
        pooled_prompt_embeds = pooled_prompt_embeds.to(torch.float8_e4m3fn)
        text_ids = text_ids.to(torch.float8_e4m3fn)

        return prompt_embeds, pooled_prompt_embeds, text_ids

class FLUXDevModel(GenericModel):
    def __init__(self, path, out_type, max_latent, steps, guidance_scale, local_path):
        super().__init__(path, out_type, max_latent, steps)
        self.local_path = local_path
        self.guidance_scale = guidance_scale

    def to(self, device):
        dtype = torch.bfloat16
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
    def __init__(self, path, out_type, max_latent, steps, revision, transformer, text_encoder_2, guidance_scale):
        super().__init__(path, out_type, max_latent, steps)
        self.revision = revision
        self.transformer = transformer
        self.text_encoder_2 = text_encoder_2
        self.guidance_scale = guidance_scale

    def to(self, device):
        dtype = torch.bfloat16
        try:
            if not self.model:
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.path, subfolder="scheduler",
                                                                            revision=self.revision)
                text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
                #text_encoder_2 = QuantizedT5EncoderModel.from_pretrained(model_name_or_path="models/flux-d/text_encoder_2")
                #text_encoder_2 = T5EncoderModel.from_pretrained("models/flux-d/text_encoder_2")
                tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype,
                                                              revision=self.revision)
                vae = AutoencoderKL.from_pretrained(self.path, subfolder="vae", torch_dtype=dtype,
                                                    revision=self.revision, device=device)
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
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.path, subfolder="scheduler",
                                                                        revision=self.revision)
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            # text_encoder_2 = QuantizedT5EncoderModel.from_pretrained(model_name_or_path="models/flux-d/text_encoder_2")
            # text_encoder_2 = T5EncoderModel.from_pretrained("models/flux-d/text_encoder_2")
            tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype,
                                                          revision=self.revision)
            vae = AutoencoderKL.from_pretrained(self.path, subfolder="vae", torch_dtype=dtype,
                                                revision=self.revision, device=device)
            self.model = FluxPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=self.transformer,
                text_encoder_2=self.text_encoder_2,
                tokenizer_2=tokenizer_2,
                vae=vae,
            )
        self.model = self.model.to(device)
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
                                          "latents"], guidance_scale=self.guidance_scale, max_sequence_length=512)
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
