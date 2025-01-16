import gc
import json
import os
import sys
import threading
import time
from typing import Dict

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel, FluxPipeline, \
    AutoencoderTiny, BitsAndBytesConfig
from diffusers.models.attention_processor import Attention, AttentionProcessor
from nunchaku.pipelines import flux as FluxSVDQPipeline
from models.T5 import QuantizedT5
from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast
import torch

from models.generic import GenericModel, GenericOutput, FinalOutput, RunStatus
from models.intermediate import IntermediateModel, IntermediateOutput

def unpack_flux_latents(latents, height, width, vae_scale_factor):
    try:
        batch_size, num_patches, channels = latents.shape

        height = 2* (int(height) // (vae_scale_factor * 2))
        width = 2* (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
        raise

    return latents


class FLUXModel(GenericModel):
    def __init__(self, path, out_type, max_latent, steps, guidance_scale, max_seq, transformerpath, res):
        super().__init__(path, out_type, max_latent, steps)
        self.guidance_scale = guidance_scale
        self.max_seq = max_seq
        self.mini_vae = "madebyollin/taef1"
        self.transformerpath = transformerpath
        self.res = res

    def to(self, device):
        dtype = torch.bfloat16
        try:
            if not self.model:
                #quantization_config = BitsAndBytesConfig(
                    #load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_skip_modules=["proj_out"]
                #    load_in_4bit=True, bnb_4bit_quant_type="nf4", llm_int8_skip_modules=["proj_out"],
                #)
                #transformer = FluxTransformer2DModel.from_pretrained(
                #    self.path,
                #    subfolder="transformer",
                #    quantization_config=quantization_config,
                #    torch_dtype=torch.bfloat16,
                #)
                #transformer = FluxTransformer2DModel.from_pretrained(self.transformerpath, torch_dtype=torch.bfloat16)
                #text_encoder_2 = QuantizedT5.from_pretrained("T5Model", torch_dtype=torch.bfloat16)
                self.model = FluxPipeline.from_pretrained(
                    self.path,
                    transformer=None,
                    #text_encoder_2=None,
                    torch_dtype=torch.bfloat16,
                )
                #self.model.text_encoder_2=text_encoder_2
        except Exception as e:
            print(repr(e))
            #transformer = FluxTransformer2DModel.from_pretrained(self.transformerpath, torch_dtype=torch.bfloat16)
            self.model = FluxPipeline.from_pretrained(
                self.path,
                transformer=None,
                torch_dtype=torch.bfloat16,
            )
        self.model.to(device)
        if device == "cuda" and self.model.transformer == None:
            self.model.transformer = FluxTransformer2DModel.from_pretrained(self.transformerpath, torch_dtype=torch.bfloat16)
        #self.model.text_encoder_2.to(device)
        #self.model.transformer.fuse_qkv_projections()
        #self.model.enable_model_cpu_offload()
        self.model.vae.enable_slicing()
        self.model.vae.enable_tiling()
        if isinstance(self.mini_vae, str):
            self.mini_vae = AutoencoderTiny.from_pretrained("madebyollin/taef1",
                                                            torch_dtype=torch.bfloat16)
        self.mini_vae.to(device)
    
    def del_model(self):
        try:
            del self.model.transformer
            del self.model
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
    
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
                                          "latents"], guidance_scale=self.guidance_scale, max_sequence_length=self.max_seq, width=self.res, height=self.res)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(repr(e))
                self.out = [[]]
                #pass
                raise e

        def progress_callback(pipe, i, t, kwargs):
            i = i + 1 # silly
            latents = kwargs["latents"]
            self.step = i
            if i % 2 and i != self.steps:
                self.intermediates = latents
                self.intermediate_update = True
            return kwargs

        for i in range(0, len(prompts), self.max_latent):
            current_prompts = prompts[i:i + self.max_latent]
            model_thread = threading.Thread(target=threaded_model,
                                            args=[[x.prompt for x in prompts[i:i + self.max_latent]],
                                                  [x.negative_prompt for x in prompts[i:i + self.max_latent]],
                                                  self.steps, progress_callback])
            model_thread.start()
            step = 0
            self.step = 0
            self.intermediates = None
            self.intermediate_update = False
            while model_thread.is_alive():
                if self.intermediate_update:
                    for idx, intermediate in enumerate(self.intermediates):
                        #no need to do this every step, so it is done during the edit limit bound stage. also gives that thread something else to do besides wait
                        #intermediate = unpack_flux_latents(intermediate.unsqueeze(0), 1024, 1024, self.model.vae_scale_factor)
                        #intermediate = ((intermediate / self.model.vae.config.scaling_factor) + self.model.vae.config.shift_factor)[0]
                        yield IntermediateOutput(output=intermediate, out_type="latent-image",
                                                 prompt=current_prompts[idx])
                    self.intermediate_update = False
                if step != self.step:
                    step = self.step
                    yield RunStatus(current=self.step,
                                    total=self.steps,
                                    interactions=[x.interaction for x in prompts[i:i + self.max_latent]])
                time.sleep(0.01)
            outputs = []
            for idx, out in enumerate(self.out[0]):
                outputs.append(
                    GenericOutput(output=out, out_type=self.out_type, prompt=prompts[i:i + self.max_latent][idx]))
            yield FinalOutput(outputs=outputs)

class FLUXSVDQModel(FLUXModel):
    def __init__(self, path, out_type, max_latent, steps, guidance_scale, max_seq, transformerpath, qmodel):
        super().__init__(path, out_type, max_latent, steps, guidance_scale, max_seq, transformerpath)
        self.qmodel = qmodel
    def to(self, device):
        dtype = torch.bfloat16
        try:
            if not self.model:
                self.model = FluxSVDQPipeline.from_pretrained(
                    self.path,
                    torch_dtype=torch.bfloat16,
                    qmodel_path=self.qmodel
                )
        except Exception as e:
            print(repr(e))
            self.model = FluxSVDQPipeline.from_pretrained(
                self.path,
                torch_dtype=torch.bfloat16,
                qmodel_path=self.qmodel
            )
        self.model.to(device)
        self.model.vae.enable_slicing()
    
    def del_model(self):
        try:
            del self.model.transformer
            del self.model
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
