from diffusers import AutoencoderTiny
from models.generic import GenericModel, GenericOutput, RunStatus, Prompt
from models.intermediate import IntermediateOutput, IntermediateOptimizedModel
from models.optimized import OptimizedModel
from diffusers.utils import numpy_to_pil
from dotenv import load_dotenv
from typing import Optional
import nextcord as discord
import threading
import asyncio
import torch
import time
import gc
import io
import os

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)
prompt_queue = []
run_queue = []
model_translations = {
    # (self, path, out_type, max_latent, steps, mini_vae)
    "sd": IntermediateOptimizedModel(path="runwayml/stable-diffusion-v1-5", out_type="image", max_latent=10, steps=25,
                                     mini_vae=AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                                                              torch_dtype=torch.float16)),
}
default_images = {
    "sd": 10
}
interaction_images = {}


class FactoryRequest:
    def __init__(self, model, prompt, negative_prompt, amount, interaction):
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.amount = amount
        self.interaction = interaction


class RunRequest:
    def __init__(self, model, prompts, negative_prompts, interactions):
        self.model = model
        self.prompts = prompts
        self.negative_prompts = negative_prompts
        self.interactions = interactions


class Output:
    def __init__(self, output, out_type, index):
        # creating the discord file here should ensure we don't unnecessarily reupload files
        imagebn = io.BytesIO
        output.save(imagebn, format="JPEG", subsampling=0, quality=90)
        imagebn.seek(0)
        self.output = discord.File(fp=imagebn, filename=str(index) + ".jpg")
        self.out_type = out_type


def model_factory():
    global prompt_queue
    global run_queue
    while True:
        while not prompt_queue:
            time.sleep(0.01)
        while len(run_queue) > 1:
            time.sleep(0.01)
        for idx, prompt in enumerate(prompt_queue):  # self, model, prompt, negative_prompt, amount, interaction
            if idx == 0:
                current_model = prompt.model
                if isinstance(current_model, str):
                    flattened_run = RunRequest(model=model_translations[prompt.model], prompts=prompt,
                                               negative_prompts=prompt.negative_prompt,
                                               interactions=[prompt.interaction])
                else:
                    flattened_run = RunRequest(model=prompt.model, prompts=prompt,
                                               negative_prompts=prompt.negative_prompt,
                                               interactions=[prompt.interaction])
            else:
                flattened_run = RunRequest(model=flattened_run.model, prompts=flattened_run.prompts + prompt,
                                           negative_prompts=flattened_run.negative_prompts + prompt.negative_prompt,
                                           interactions=flattened_run.interactions + prompt.interaction)
        last_interaction = None
        for interaction in flattened_run.interactions:
            if interaction != last_interaction:
                asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message(content="Model loading..."),
                                                 loop=client.loop)
            last_interaction = interaction
        flattened_run.model.to("cpu")
        if isinstance(flattened_run.model, IntermediateOptimizedModel):
            flattened_run.mini_vae.to("cpu")
        for interaction in flattened_run.interactions:
            if interaction != last_interaction:
                asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message(content="Model loaded to cpu"),
                                                 loop=client.loop)
            last_interaction = interaction
        run_queue.append(flattened_run)
        prompt_queue.pop(0)

async def async_model_runner():
    global run_queue
    global interaction_images
    while True:
        if not run_queue:
            time.sleep(0.01)
        now = run_queue[
            0]  # this is a list of FactoryRequests. self, model, prompt, negative_prompt, amount, interaction
        prompts = []
        images = {}
        now.model.to('cuda')
        for request in now:
            for i in range(request.amount):
                prompts.append(Prompt(prompt=request.prompt, negative_prompt=request.negative_prompt,
                                      interaction=request.interaction, index=i))
                if not request.interaction in images:
                    images[request.interaction] = [None] * request.amount
                asyncio.run_coroutine_threadsafe(
                    coro=request.interaction.edit_original_message(content="Model loaded to gpu"), loop=client.loop)
        limiter = time.time()
        async for i in now.model.call(prompts):
            if type(i) == GenericOutput:  #(self, output, out_type, interaction, index)
                #This event is final, meaning this image is done.
                images[i.interaction][i.index] = Output(output=i.output, out_type=i.out_type, index=i.index)
            if type(i) == IntermediateOutput:
                #output = i.output.to('cpu', non_blocking=True)
                images[i.interaction][i.index] = Output(output=i.output, out_type=i.out_type, index=i.index)
            if type(i) == RunStatus:
                interactions = list(set(i.interactions))
                for interaction in interactions:
                    sendable_images = []
                    for image in images[interaction]:
                        if image != None:
                            if image.out_type == 'latents':
                                image = image.output.to('cpu', non_blocking=True)
                                image = numpy_to_pil((image / 2 + 0.5).permute(1, 2, 0).numpy())[0].resize((256, 256))
                            sendable_images.append(image)
                    if sendable_images:
                        progress = len(sendable_images) * i.current / i.total
                        asyncio.run_coroutine_threadsafe(
                            coro=interaction.edit_original_message(content=str(round(progress, 2)) + "%",
                                                                   files=sendable_images), loop=client.loop)
        for request in now:
            sendable_images = []
            for image in images[request.interaction]:
                if image != None: sendable_images.append(image)
            if sendable_images != []:
                if now.negative_prompt != "":
                    asyncio.run_coroutine_threadsafe(coro=request.interaction.edit_original_message(content=str(
                        request.amount) + " images of '" + request.prompt + "' + (negative: '" + request.negative_prompt + "'",
                                                                                                    files=sendable_images),
                                                     loop=client.loop)
                else:
                    asyncio.run_coroutine_threadsafe(coro=request.interaction.edit_original_message(
                        content=str(request.amount) + " images of '" + request.prompt + "'", files=sendable_images),
                        loop=client.loop)
        model_reused = False
        if len(run_queue) > 1:
            if run_queue[1].model.path == now.model.path:
                run_queue[1].model = now.model.to('cpu')
                model_reused = True
            elif len(run_queue) > 2:
                if run_queue[2].model.path == now.model.path:
                    run_queue[2].model = now.model.to('cpu')
                    model_reused = True
        if not model_reused:
            del now.model
        gc.collect()
        torch.cuda.empty_cache()
        run_queue.pop(0)


def model_runner():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_model_runner)


@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')


@client.slash_command(description="Generates an image from the prompt")
async def generate(
        interaction: discord.Interaction,
        prompt: str = discord.SlashOption(
            name="prompt",
            required=True,
            description="The prompt to generate off of",
            max_length=1024,
        ),
        negative_prompt: Optional[str] = discord.SlashOption(
            name="negative_prompt",
            required=True,
            description="The negative prompt to generate off of",
            max_length=1024,
        ),
        model: Optional[str] = discord.SlashOption(
            name="model",
            choices=["sd"],
            required=False,
            description="The model to use to generate the image",
        ),
        images: Optional[int] = discord.SlashOption(
            name="images",
            choices={"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10},
            required=False,
            description="How many images to generate (more will take longer)"
        ),
):
    global default_images
    global prompt_queue
    if not model: model = "sd"
    if not images: images = default_images[model]
    if not negative_prompt: negative_prompt = None
    interaction.response.send_message("Generation has been queued.")
    # (self, model, prompt, negative_prompt, amount, interaction)
    prompt_queue.append(FactoryRequest(model=model, prompt=prompt, negative_prompt=negative_prompt, amount=images,
                                       interaction=interaction))


threading.Thread(target=model_factory, daemon=True).start()
client.run(TOKEN)
