import gc
import json
import os
import sys
import threading
import time

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel, FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast, T5EncoderModel
from optimum.quanto import freeze, quantize, qint8
import torch

from models.generic import GenericModel, GenericOutput, FinalOutput, RunStatus

#from accelerate import infer_auto_device_map

class FLUXDevModel(GenericModel):
    def __init__(self, path, out_type, max_latent, steps, revision):
        super().__init__(path, out_type, max_latent, steps)
        self.revision = revision

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
                text_encoder_2 = T5EncoderModel.from_pretrained(self.path, subfolder="text_encoder_2",
                                                                torch_dtype=dtype,
                                                                revision=self.revision)
                tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype,
                                                              revision=self.revision)
                vae = AutoencoderKL.from_pretrained(self.path, subfolder="vae", torch_dtype=dtype,
                                                    revision=self.revision, device=device)
                #transformer = FluxTransformer2DModel.from_pretrained("models/flux-d/transformer")
                transformer = FluxTransformer2DModel.from_pretrained(self.path, subfolder="transformer", revision=self.revision, torch_dtype=dtype)
                #transformer = FluxTransformer2DModel.from_single_file("models/flux-d/flux1-dev-fp8.safetensors", torch_dtype=torch.float8_e4m3fn)
                #transformer_device_map = infer_auto_device_map(transformer, max_memory={0: "19GiB", "cpu": "64GiB"})
                #print(transformer_device_map)
                loader_threads = []
                def quantize_transformer():
                    quantize(transformer, qint8)
                    freeze(transformer)
                    print("Finished quantizing transformer")
                def quantize_text_encoder_2():
                    quantize(text_encoder_2, qint8)
                    freeze(text_encoder_2)
                    print("Finished quantizing text encoder")
                loader_threads.append(threading.Thread(target=quantize_transformer))
                loader_threads.append(threading.Thread(target=quantize_text_encoder_2))
                for thread in loader_threads:
                    thread.start()
                for thread in loader_threads:
                    thread.join()
                #text_encoder_2.to(device=device)
                #transformer.to(device=device)
                self.model = FluxPipeline(
                    scheduler=scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=transformer,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                )
                #self.model.text_encoder_2 = text_encoder_2
                #self.model.transformer = transformer
        except:
            gc.collect()
            torch.cuda.empty_cache()
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.path, subfolder="scheduler",
                                                                        revision=self.revision)
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            # text_encoder_2 = QuantizedT5EncoderModel.from_pretrained(model_name_or_path="models/flux-d/text_encoder_2")
            # text_encoder_2 = T5EncoderModel.from_pretrained("models/flux-d/text_encoder_2")
            text_encoder_2 = T5EncoderModel.from_pretrained(self.path, subfolder="text_encoder_2",
                                                            torch_dtype=dtype,
                                                            revision=self.revision)
            tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype,
                                                          revision=self.revision)
            vae = AutoencoderKL.from_pretrained(self.path, subfolder="vae", torch_dtype=dtype,
                                                revision=self.revision, device=device)
            # transformer = FluxTransformer2DModel.from_pretrained("models/flux-d/transformer")
            transformer = FluxTransformer2DModel.from_pretrained(self.path, subfolder="transformer",
                                                                 revision=self.revision, torch_dtype=dtype)
            # transformer = FluxTransformer2DModel.from_single_file("models/flux-d/flux1-dev-fp8.safetensors", torch_dtype=torch.float8_e4m3fn)
            # transformer_device_map = infer_auto_device_map(transformer, max_memory={0: "19GiB", "cpu": "64GiB"})
            # print(transformer_device_map)
            loader_threads = []

            def quantize_transformer():
                quantize(transformer, qint8)
                freeze(transformer)
                print("Finished quantizing transformer")

            def quantize_text_encoder_2():
                quantize(text_encoder_2, qint8)
                freeze(text_encoder_2)
                print("Finished quantizing text encoder")

            loader_threads.append(threading.Thread(target=quantize_transformer))
            loader_threads.append(threading.Thread(target=quantize_text_encoder_2))
            for thread in loader_threads:
                thread.start()
            for thread in loader_threads:
                thread.join()
            # text_encoder_2.to(device=device)
            # transformer.to(device=device)
            self.model = FluxPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                vae=vae,
            )
            # self.model.text_encoder_2 = text_encoder_2
            # self.model.transformer = transformer
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
    def __init__(self, path, out_type, max_latent, steps, revision, transformer, text_encoder_2):
        super().__init__(path, out_type, max_latent, steps)
        self.revision = revision
        self.transformer = transformer
        self.text_encoder_2 = text_encoder_2

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
