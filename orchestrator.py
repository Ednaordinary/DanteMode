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


class Started:
    def __init__(self, name):
        self.name = name

class Loaded:
    def __init__(self, name):
        self.name = name

def main(queue):
    app = modal.App("Dante")
    image = modal.Image \
        .from_registry("pytorch/pytorch:latest") \
        .apt_install("git", "python3", "python-is-python3") \
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
        .pip_install("huggingface_hub", "torch", "numpy")

    @app.function(image=slim_image, volumes={"/model": volume}, secrets=[hf_token], serialized=True, retries=0, memory=1024*1, timeout=60*60)
    def download(model, path, ignore, **kwargs):
        print("Importing downloader")
        import torch
        from huggingface_hub import snapshot_download
        print("Starting download")
        ignore = ["*.pt", "*.bin"]
        ignore.extend(ignore)
        snapshot_download(
            model,
            local_dir="/model/" + path,
            ignore_patterns=ignore,  # using safetensors
        )
        print("Download complete")
        volume.commit()

    @app.function(gpu="T4", image=image, volumes={"/model": volume}, serialized=True, retries=0)
    async def get_embeds(wrapper, queue, **kwargs):
        queue.put(Started())
        wrapper = wrapper()
        wrapper.load()
        queue.put(Loaded())
        embeds = wrapper.embeds()


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
