import threading
import time

from DeepCache import DeepCacheSDHelper

from .generic import GenericOutput, FinalOutput, RunStatus, GenericModel
from .intermediate import IntermediateOptimizedModel, IntermediateModel, IntermediateOutput
from diffusers import AutoencoderKL, AutoencoderTiny, DiffusionPipeline, DPMSolverMultistepScheduler, \
    AutoPipelineForText2Image, StableDiffusion3Pipeline, StableCascadeDecoderPipeline, StableCascadePriorPipeline, \
    DEISMultistepScheduler
import torch
import gc


class SDXLModel(IntermediateOptimizedModel):
    def to(self, device):
        try:
            if not self.model:
                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
                self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                               safety_checker=None,
                                                               vae=vae, variant="fp16", use_safetensors=True)
                del vae
        except:
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None,
                                                           vae=vae, variant="fp16", use_safetensors=True)
            del vae
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config,
                                                                       use_karras_sigmas=True)  # , use_lu_lambdas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.helper = DeepCacheSDHelper(pipe=self.model)
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)


class SDDSModel(IntermediateOptimizedModel):
    def to(self, device):
        try:
            if not self.model:
                self.model = AutoPipelineForText2Image.from_pretrained(self.path, torch_dtype=torch.float16,
                                                                       variant="fp16", safety_checker=None,
                                                                       use_safetensors=True)
                self.model.scheduler = DEISMultistepScheduler.from_config(self.model.scheduler.config)
        except:
            self.model = AutoPipelineForText2Image.from_pretrained(self.path, torch_dtype=torch.float16,
                                                                   variant="fp16", safety_checker=None,
                                                                   use_safetensors=True)
            self.model.scheduler = DEISMultistepScheduler.from_config(self.model.scheduler.config)
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()
        self.helper = DeepCacheSDHelper(pipe=self.model)
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)


class SDXLDSModel(IntermediateOptimizedModel):
    def to(self, device):
        try:
            if not self.model:
                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
                self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                               safety_checker=None,
                                                               vae=vae, variant="fp16", use_safetensors=True)
                del vae
        except:
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None,
                                                           vae=vae, variant="fp16", use_safetensors=True)
            del vae
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()
        self.helper = DeepCacheSDHelper(pipe=self.model)
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)


class SDXLJXModel(IntermediateOptimizedModel):
    def to(self, device):
        try:
            if not self.model:
                vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
                self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                               safety_checker=None,
                                                               vae=vae)
                del vae
        except:
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            self.model = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None,
                                                           vae=vae)
            del vae
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config,
                                                                       use_karras_sigmas=True)  # , use_lu_lambdas=True)
        self.model.scheduler.algorithm_type = "dpmsolver++"
        self.helper = DeepCacheSDHelper(pipe=self.model)
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
        self.mini_vae.to(device)


class SDXLTModel(GenericModel):
    def to(self, device):
        try:
            self.model
        except:
            self.model = AutoPipelineForText2Image.from_pretrained(self.path, torch_dtype=torch.float16,
                                                                   safety_checker=None, variant="fp16",
                                                                   use_safetensors=True)
        else:
            if not self.model:
                self.model = AutoPipelineForText2Image.from_pretrained(self.path, torch_dtype=torch.float16,
                                                                       safety_checker=None, variant="fp16",
                                                                       use_safetensors=True)
        if self.model.device.type != device:
            self.model = self.model.to(device)
        self.model.vae.enable_slicing()

    async def call(self, prompts):
        if self.model.device.type != "cuda":
            self.to("cuda")
        for i in range(0, len(prompts), self.max_latent):
            #For SDXL Turbo we don't bother with run status's, it's too fast and just rate limits us
            try:
                self.out = self.model([x.prompt for x in prompts[i:i + self.max_latent]],
                                      negative_prompt=[x.negative_prompt for x in prompts[i:i + self.max_latent]],
                                      num_inference_steps=self.steps, guidance_scale=0.0)
            except:
                self.out = [[]]
            outputs = []
            for idx, out in enumerate(self.out[0]):
                outputs.append(
                    GenericOutput(output=out, out_type=self.out_type, prompt=prompts[i:i + self.max_latent][idx]))
            yield FinalOutput(outputs=outputs)


class SD3Model(IntermediateModel):
    def to(self, device):
        try:
            if not self.model:
                self.model = StableDiffusion3Pipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                                      safety_checker=None, use_safetensors=True)
        except:
            self.model = StableDiffusion3Pipeline.from_pretrained(self.path, torch_dtype=torch.float16,
                                                                  safety_checker=None, use_safetensors=True)
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained(self.mini_vae,
                                                            torch_dtype=torch.float16)
            self.mini_vae.config.shift_factor = 0.0
        self.mini_vae.to(device)

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
                                      callback_on_step_end_tensor_inputs=["latents"],
                                      max_sequence_length=512)  # callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
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


class SCASCModel(GenericModel):
    def to(self, device):
        try:
            if not self.model:
                self.prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior",
                                                                        variant="bf16",
                                                                        torch_dtype=torch.bfloat16, safety_checker=None)
                self.model = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16",
                                                                          torch_dtype=torch.float16,
                                                                          safety_checker=None)
        except:
            self.prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16",
                                                                    torch_dtype=torch.bfloat16, safety_checker=None)
            self.model = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16",
                                                                      torch_dtype=torch.float16, safety_checker=None)
        self.prior = self.prior.to(device)
        self.model = self.model.to(device)

    def del_model(self):
        del self.prior
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    async def call(self, prompts):
        self.to("cuda")

        def intermediate_callback_prior(pipe, i, t, kwargs):
            self.prior_step = i
            self.intermediate_update = True
            return kwargs

        def intermediate_callback(pipe, i, t, kwargs):
            self.step = i
            self.intermediate_update = True
            return kwargs

        def threaded_model(prompts, negative_prompts):
            try:
                #self.out = self.model(prompts, negative_prompt=[x if x != None else "" for x in negative_prompts],
                #                      num_inference_steps=steps, callback_on_step_end=callback)  # callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
                self.prior.to("cuda")
                self.out = self.prior(prompts, negative_prompt=[x if x != None else "" for x in negative_prompts],
                                      guidance_scale=4.0, num_inference_steps=self.steps, height=1024, width=1024,
                                      callback_on_step_end=intermediate_callback_prior)
                embeddings = self.out.image_embeddings.to(torch.float16)
                self.prior.to("cpu")  # Prior cant be on cuda during the decoding step, otherwise we run out of memory
                self.out = self.model(image_embeddings=embeddings, prompt=prompts,
                                      negative_prompt=[x if x != None else "" for x in negative_prompts],
                                      guidance_scale=0.0, output_type="pil", num_inference_steps=self.steps,
                                      callback_on_step_end=intermediate_callback).images
            except Exception as e:
                print(repr(e))
                self.out = [[]]
                pass

        for im in range(0, len(prompts), self.max_latent):
            current_prompts = prompts[im:im + self.max_latent]
            model_thread = threading.Thread(target=threaded_model,
                                            args=[[x.prompt for x in prompts[im:im + self.max_latent]],
                                                  [x.negative_prompt for x in prompts[im:im + self.max_latent]]])
            model_thread.start()
            self.intermediate_update = False
            self.prior_step = 0
            self.step = 0
            while model_thread.is_alive():
                if self.intermediate_update:
                    yield RunStatus(current=(self.step + self.prior_step) / 2,
                                    total=self.steps,
                                    interactions=[x.interaction for x in prompts[im:im + self.max_latent]])
                    self.intermediate_update = False
                time.sleep(0.01)
            outputs = []
            for idx, out in enumerate(self.out):
                outputs.append(GenericOutput(output=out, out_type=self.out_type,
                                             prompt=current_prompts[idx]))
            yield FinalOutput(outputs=outputs)
            self.intermediates = None
            self.step = 0
            gc.collect()
            torch.cuda.empty_cache()
