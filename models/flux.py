import gc
import threading
import time

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel, FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from optimum.quanto import freeze, qfloat8, quantize
import torch

from models.generic import GenericModel, GenericOutput, FinalOutput, RunStatus


class FLUXModel(GenericModel):
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
                # text_encoder_2 = T5EncoderModel.from_pretrained(self.path, subfolder="text_encoder_2", torch_dtype=dtype,
                #                                                 revision=self.revision)
                # tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype,
                #                                               revision=self.revision)
                vae = AutoencoderKL.from_pretrained(self.path, subfolder="vae", torch_dtype=dtype, revision=self.revision)
                transformer = FluxTransformer2DModel.from_pretrained(self.path, subfolder="transformer",
                                                                     torch_dtype=dtype, revision=self.revision)
                quantize(transformer, weights=qfloat8)
                freeze(transformer)
                # quantize(text_encoder_2, weights=qfloat8)
                # freeze(text_encoder_2)
                self.model = FluxPipeline(
                    scheduler=scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=transformer,
                    # text_encoder_2=text_encoder_2,
                    # tokenizer_2=tokenizer_2,
                    text_encoder_2=None,
                    tokenizer_2=None,
                    vae=vae,
                )
        except:
            gc.collect()
            torch.cuda.empty_cache()
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.path, subfolder="scheduler",
                                                                        revision=self.revision)
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
            # text_encoder_2 = T5EncoderModel.from_pretrained(self.path, subfolder="text_encoder_2", torch_dtype=dtype,
            #                                                 revision=self.revision)
            # tokenizer_2 = T5TokenizerFast.from_pretrained(self.path, subfolder="tokenizer_2", torch_dtype=dtype,
            #                                               revision=self.revision)
            vae = AutoencoderKL.from_pretrained(self.path, subfolder="vae", torch_dtype=dtype, revision=self.revision)
            transformer = FluxTransformer2DModel.from_pretrained(self.path, subfolder="transformer",
                                                                 torch_dtype=dtype, revision=self.revision)
            quantize(transformer, weights=qfloat8)
            freeze(transformer)
            # quantize(text_encoder_2, weights=qfloat8)
            # freeze(text_encoder_2)
            self.model = FluxPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
                # text_encoder_2=text_encoder_2,
                # tokenizer_2=tokenizer_2,
                text_encoder_2=None,
                tokenizer_2=None,
                vae=vae,
            )
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()

    async def call(self, prompts):
        self.to("cuda")

        def threaded_model(model, prompts, negative_prompts, steps, callback):
            try:
                #Flux in diffusers doesnt support negative_prompt rn :(
                self.out = model(prompts, num_inference_steps=steps,
                                 callback_on_step_end=callback,
                                 callback_on_step_end_tensor_inputs=[
                                     "latents"])
            except Exception as e:
                print(repr(e))
                self.out = [[]]
                pass

        def progress_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            self.step = i
            return kwargs

        for i in range(0, len(prompts), self.max_latent):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[self.model, [x.prompt for x in prompts[i:i + self.max_latent]],
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