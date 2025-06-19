import modal
import asyncio

app = modal.App("Dante")
image = modal.Image \
    .from_registry("pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime") \
    .pip_install(
    "git+https://github.com/huggingface/diffusers",
    "transformers",
    "accelerate",
    "sentencepiece",
    "protobuf",
    "hf_transfer",
) \
    .pip_install(
    "flash-attn", extra_options="--no-build-isolation"
) \
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
volume = modal.Volume.from_name("dante-models", create_if_missing=True)
model_dir = "/model"
hf_token = modal.Secret.from_name(
    "hf-token", required_keys=["HF_TOKEN"]
)


@app.function(image=image, volumes={"/model": volume}, secrets=[hf_token])
async def download(self, path, kwargs):
    from diffusers import DiffusionPipeline
    DiffusionPipeline.from_pretrained(path, **kwargs)


@app.function(gpu="T4", image=image, volumes={"/model": volume})
async def get_embeds(self, pipeline):
    pass


app.run()
