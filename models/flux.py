import gc
import json
import os
import sys
import threading
import time
from typing import Union, Any, Optional, List

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel, FluxPipeline
from diffusers.utils import is_accelerate_available
from optimum.quanto.models.shared_dict import ShardedStateDict
from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast, T5EncoderModel, AutoConfig, PreTrainedModel
from optimum.quanto.models.diffusers_models import QuantizedDiffusersModel
from optimum.quanto.models.transformers_models import QuantizedTransformersModel
from optimum.quanto import freeze, quantize, qint8, requantize, quantization_map, qtype, Optimizer, QModuleMixin
import torch
from transformers.modeling_utils import load_state_dict
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from transformers.utils.hub import get_checkpoint_shard_files

from models.generic import GenericModel, GenericOutput, FinalOutput, RunStatus


class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel


# Class from https://github.com/huggingface/optimum-quanto/blob/main/optimum/quanto/models/transformers_models.py
class LoadTweakedQuantizedTransformersModel(QuantizedTransformersModel):
    BASE_NAME = "quanto"
    auto_class = None

    def __init__(self, model: PreTrainedModel):
        if not isinstance(model, PreTrainedModel) or len(quantization_map(model)) == 0:
            raise ValueError("The source model must be a quantized transformers model.")
        self._wrapped = model

    def __getattr__(self, name: str) -> Any:
        """If an attribute is not found in this class, look in the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = self.__dict__["_wrapped"]
            return getattr(wrapped, name)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    @staticmethod
    def _qmap_name():
        return f"{QuantizedTransformersModel.BASE_NAME}_qmap.json"

    @classmethod
    def quantize(
            cls,
            model: PreTrainedModel,
            weights: Optional[Union[str, qtype]] = None,
            activations: Optional[Union[str, qtype]] = None,
            optimizer: Optional[Optimizer] = None,
            include: Optional[Union[str, List[str]]] = None,
            exclude: Optional[Union[str, List[str]]] = None,
    ):
        """Quantize the specified model

        By default, each layer of the model will be quantized if is quantizable.

        If include patterns are specified, the layer name must match one of them.

        If exclude patterns are specified, the layer must not match one of them.

        Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
        https://docs.python.org/3/library/fnmatch.html for more details.

        Note: quantization happens in-place and modifies the original model.

        Note that the resulting quantized model will be frozen: if you wish to do
        quantization-aware training then you should use `optimum.quanto.quantize` instead,
        and call `optimum.quanto.freeze` only after the training.

        Args:
            model (`PreTrainedModel`): the model to quantize.
            weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
            activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
            include (`Optional[Union[str, List[str]]]`):
                Patterns constituting the allowlist. If provided, layer names must match at
                least one pattern from the allowlist.
            exclude (`Optional[Union[str, List[str]]]`):
                Patterns constituting the denylist. If provided, layer names must not match
                any patterns from the denylist.
        """
        if not isinstance(model, PreTrainedModel):
            raise ValueError("The source model must be a transformers model.")
        quantize(
            model, weights=weights, activations=activations, optimizer=optimizer, include=include, exclude=exclude
        )
        freeze(model)
        return cls(model)

    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, os.PathLike]):
        if cls.auto_class is None:
            raise ValueError(
                "Quantized models cannot be reloaded using {cls}: use a specialized quantized class such as QuantizedModelForCausalLM instead."
            )
        if not is_accelerate_available():
            raise ValueError("Reloading a quantized transformers model requires the accelerate library.")
        from accelerate import init_empty_weights

        if os.path.isdir(model_name_or_path):
            # Look for a quantization map
            qmap_path = os.path.join(model_name_or_path, cls._qmap_name())
            if not os.path.exists(qmap_path):
                raise ValueError(f"No quantization map found in {model_name_or_path}: is this a quantized model ?")
            with open(qmap_path, "r", encoding="utf-8") as f:
                qmap = json.load(f)
            # Create an empty model
            config = AutoConfig.from_pretrained(model_name_or_path)
            with init_empty_weights():
                model = cls.auto_class(config)
            # Look for the index of a sharded checkpoint
            checkpoint_file = os.path.join(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
            if os.path.exists(checkpoint_file):
                # Convert the checkpoint path to a list of shards
                checkpoint_file, sharded_metadata = get_checkpoint_shard_files(model_name_or_path, checkpoint_file)
                # Create a mapping for the sharded safetensor files
                state_dict = ShardedStateDict(model_name_or_path, sharded_metadata["weight_map"])
            else:
                # Look for a single checkpoint file
                checkpoint_file = os.path.join(model_name_or_path, SAFE_WEIGHTS_NAME)
                if not os.path.exists(checkpoint_file):
                    raise ValueError(f"No safetensor weights found in {model_name_or_path}.")
                # Get state_dict from model checkpoint
                state_dict = load_state_dict(checkpoint_file)
            # Requantize and load quantized weights from state_dict
            requantize(model, state_dict=state_dict, quantization_map=qmap)
            if getattr(model.config, "tie_word_embeddings", True):
                # Tie output weight embeddings to input weight embeddings
                # Note that if they were quantized they would NOT be tied
                model.tie_weights()
            # Set model in evaluation mode as it is done in transformers
            model.eval()
            return cls(model)
        else:
            raise NotImplementedError("Reloading quantized models directly from the hub is not supported yet.")

    def save_pretrained(self, save_directory: Union[str, os.PathLike], max_shard_size: Union[int, str] = "5GB"):

        model = self._wrapped
        if getattr(model.config, "tie_word_embeddings", True):
            # The original model had tied embedding inputs and outputs
            if isinstance(model.get_input_embeddings(), QModuleMixin) or isinstance(
                    model.get_output_embeddings(), QModuleMixin
            ):
                # At least one of the two is quantized, so they are not tied anymore
                model.config.tie_word_embeddings = False
        self._wrapped.save_pretrained(save_directory, max_shard_size=max_shard_size, safe_serialization=True)
        # Save quantization map to be able to reload the model
        qmap_name = os.path.join(save_directory, self._qmap_name())
        qmap = quantization_map(self._wrapped)
        with open(qmap_name, "w", encoding="utf8") as f:
            json.dump(qmap, f, indent=4)


class QuantizedT5EncoderModel(LoadTweakedQuantizedTransformersModel):
    auto_class = T5EncoderModel


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
                                                    revision=self.revision)
                #transformer = FluxTransformer2DModel.from_pretrained("models/flux-d/transformer")
                transformer = FluxTransformer2DModel.from_pretrained(self.path, subfolder="transformer",
                                                                     torch_dtype=dtype, revision=self.revision)
                quantize(transformer, qint8)
                freeze(transformer)
                #text_encoder_2.to(device=device)
                #transformer.to(device=device)
                self.model = FluxPipeline(
                    scheduler=scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=None,
                    text_encoder_2=None,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                )
                self.model.text_encoder_2 = text_encoder_2
                self.model.transformer = transformer
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
                                                revision=self.revision)
            # transformer = FluxTransformer2DModel.from_pretrained("models/flux-d/transformer")
            transformer = FluxTransformer2DModel.from_pretrained(self.path, subfolder="transformer",
                                                                 torch_dtype=dtype, revision=self.revision)
            quantize(transformer, qint8)
            freeze(transformer)
            # text_encoder_2.to(device=device)
            # transformer.to(device=device)
            self.model = FluxPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=None,
                text_encoder_2=None,
                tokenizer_2=tokenizer_2,
                vae=vae,
            )
            self.model.text_encoder_2 = text_encoder_2
            self.model.transformer = transformer
        self.model = self.model.to(device)
        #self.model.enable_model_cpu_offload()
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
