import modal
import threading
from queue import Queue


class Function:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs


class End:
    def __init__(self, queue):
        self.queue = queue


def main(queue):
    app = modal.App("Dante")
    image = modal.Image \
        .from_registry("nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04") \
        .apt_install("git") \
        .pip_install(
        "git+https://github.com/huggingface/diffusers",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "hf_transfer",
        "torch",
        "numpy",
    ) \
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    volume = modal.Volume.from_name("dante-models", create_if_missing=True)
    model_dir = "/model"
    hf_token = modal.Secret.from_name(
        "hf-token", required_keys=["HF_TOKEN"]
    )

    slim_image = modal.Image \
        .debian_slim() \
        .pip_install("huggingface_hub", "torch")

    @app.function(image=slim_image, volumes={"/model": volume}, secrets=[hf_token], serialized=True, retries=0, memory=1024*1, timeout=60*60)
    def download(model, path, **kwargs):
        print("Importing downloader")
        import torch
        from huggingface_hub import snapshot_download
        print("Starting download")
        snapshot_download(
            model,
            local_dir="/model/" + path,
            ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        )
        print("Download complete")
        volume.commit()

    @app.function(gpu="T4", image=image, volumes={"/model": volume}, serialized=True, retries=0)
    async def get_embeds(pipeline):
        pass

    def wait_for_end(queue, calls):
        for i in calls:
            print(i.get())
        queue.put(1)

    with modal.enable_output():
        with app.run():
            calls = []
            while True:
                item: Function = queue.get()
                if isinstance(item, End):
                    wait_for_end(item.queue, calls)
                    break
                calls.append(locals()[item.function].spawn(*item.args, **item.kwargs))


def start():
    queue = Queue()
    threading.Thread(target=main, args=[queue]).start()
    return queue
