import gc
import os
import sys
import threading
import time

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel, FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from optimum.quanto.models.diffusers_models import QuantizedDiffusersModel
from optimum.quanto.models.transformers_models import QuantizedModelForCausalLM
from optimum.quanto import freeze, quantize, qint8
import torch

from models.generic import GenericModel, GenericOutput, FinalOutput, RunStatus

class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):

    base_class = FluxTransformer2DModel

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
                text_encoder_2 = QuantizedModelForCausalLM.from_pretrained("models/flux-d/text_encoder_2")
                #text_encoder_2 = T5EncoderModel.from_pretrained("models/flux-d/text_encoder_2")
                tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype,
                                                              revision=self.revision)
                vae = AutoencoderKL.from_pretrained(self.path, subfolder="vae", torch_dtype=dtype, revision=self.revision)
                #transformer = FluxTransformer2DModel.from_pretrained("models/flux-d/transformer")
                transformer = QuantizedFluxTransformer2DModel.from_pretrained("models/flux-d/transformer")
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
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.path, subfolder="scheduler",
                                                                        revision=self.revision)
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            text_encoder_2 = QuantizedModelForCausalLM.from_pretrained("models/flux-d/text_encoder_2")
            tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype,
                                                          revision=self.revision)
            vae = AutoencoderKL.from_pretrained(self.path, subfolder="vae", torch_dtype=dtype, revision=self.revision)
            #transformer = FluxTransformer2DModel.from_pretrained("models/flux-d/transformer")
            transformer = QuantizedFluxTransformer2DModel.from_pretrained("models/flux-d/transformer")
            self.model = FluxPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                vae=vae,
            )
        if device != "cuda":
            self.model = self.model.to(device)
        else:
            self.model.enable_model_cpu_offload()
        self.model.vae.enable_slicing()
        print(self.model)

    async def call(self, prompts):
        self.to("cuda")

        def threaded_model(prompts, negative_prompts, steps, callback):
            try:
                #Flux in diffusers doesnt support negative_prompt rn :(
                print(self.model)
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
                pass

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