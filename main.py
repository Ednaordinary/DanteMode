import sys

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
    "sd": IntermediateOptimizedModel(path="runwayml/stable-diffusion-v1-5", out_type="image", max_latent=20, steps=25,
                                     mini_vae="madebyollin/taesd"),
}
default_images = {
    "sd": 10
}
images = {}

class UpscaleAndAgainAndEdit(discord.ui.View):
    def __init__(self, *, timeout=None, prompt):
        super().__init__(timeout=timeout)
        self.prompt = prompt
    @discord.ui.button(label="Again", style=discord.ButtonStyle.primary)
    async def again_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        prompt_queue.append()
        button.style = discord.ButtonStyle.secondary

class FactoryRequest:
    def __init__(self, model, prompt, negative_prompt, amount, interaction):
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.amount = amount
        self.interaction = interaction


#class RunRequest:
#    def __init__(self, model, prompts, negative_prompts, interactions):
#        self.model = model
#        self.prompts = prompts
#        self.negative_prompts = negative_prompts
#        self.interactions = interactions


class Output:
    def __init__(self, output, out_type, index):
        self.output = output
        self.out_type = out_type
        self.index = index

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
                    prompt.model = model_translations[prompt.model]
                    flattened_run = [prompt]
                else:
                    flattened_run = [prompt]
            else:
                if prompt.model == current_model:
                    flattened_run.append(prompt)
        last_interaction = None
        for interaction in [x.interaction for x in flattened_run]:
            if interaction != last_interaction:
                asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message(content="Model loading..."),
                                                 loop=client.loop)
            last_interaction = interaction
        model_already_loaded = False
        if len(run_queue) > 0:
            if run_queue[0][0].model.path == flattened_run[0].model.path:
                model_already_loaded = True
        if not model_already_loaded:
            flattened_run[0].model.to("cpu")
        for interaction in [x.interaction for x in flattened_run]:
            if interaction != last_interaction:
                asyncio.run_coroutine_threadsafe(
                    coro=interaction.edit_original_message(content="Model loaded to gpu" if model_already_loaded else "Model loaded to gpu"),
                    loop=client.loop)
            last_interaction = interaction
        run_queue.append(flattened_run)
        prompt_queue.pop(0)

async def async_model_runner():
    global run_queue
    global images
    while True:
        while len(run_queue) == 0:
            time.sleep(0.01)
        now = run_queue[0]
        # this is a list of FactoryRequests. self, model, prompt, negative_prompt, amount, interaction
        prompts = []
        now[0].model.to('cuda')
        for request in now:
            for i in range(request.amount):
                prompts.append(Prompt(prompt=request.prompt, negative_prompt=request.negative_prompt,
                                      interaction=request.interaction, index=i))
                if not request.interaction in images:
                    images[request.interaction] = [None] * request.amount
                asyncio.run_coroutine_threadsafe(
                    coro=request.interaction.edit_original_message(content="Model loaded to gpu"), loop=client.loop)
        limiter = time.time()
        async for i in now[0].model.call(prompts):
            if type(i) == GenericOutput:  #(self, output, out_type, interaction, index)
                #This event is final, meaning this image is done.
                images[i.interaction][i.index] = None
                gc.collect()
                torch.cuda.empty_cache()
                images[i.interaction][i.index] = Output(output=i.output, out_type=i.out_type[0], index=i.index)
            if type(i) == IntermediateOutput:
                #output = i.output.to('cpu', non_blocking=True)
                images[i.interaction][i.index] = None
                gc.collect()
                torch.cuda.empty_cache()
                images[i.interaction][i.index] = Output(output=i.output, out_type=i.out_type[0], index=i.index)
            if type(i) == RunStatus:
                if limiter + 1.0 < time.time():
                    interactions = list(set(i.interactions))
                    for interaction in interactions:
                        sendable_images = []
                        current = 0
                        already_done = 0
                        for idx, image in enumerate(images[interaction]):
                            if image != None:
                                if image.out_type == 'latent-image':
                                    image = image.output.to('cpu', non_blocking=True)
                                    image = numpy_to_pil((image / 2 + 0.5).permute(1, 2, 0).numpy())[0].resize((128, 128))
                                    with io.BytesIO() as imagebn:
                                        image.save(imagebn, format="JPEG", quality=50)
                                        imagebn.seek(0)
                                        sendable_images.append(discord.File(fp=imagebn, filename=str(idx) + ".jpg"))
                                    current += 1
                                else:
                                    with io.BytesIO() as imagebn:
                                        image.output.save(imagebn, format="JPEG", subsampling=0, quality=90)
                                        imagebn.seek(0)
                                        sendable_images.append(discord.File(fp=imagebn, filename=str(image.index) + ".jpg"))
                                    already_done += 1
                        if sendable_images:
                            for request in now:
                                if request.interaction == interaction:
                                    this_request = request
                                    break
                            # separating this because it is CONFUSING
                            #current = i.current * len([x for x in i.interactions if x == interaction])
                            #already_done = len([x for x in images[interaction] if x.out_type == "image"]) * i.total[0]
                            print(current, already_done, i.total[0], request.amount)
                            #progress = 100 * (current + already_done) / (i.total[0] * request.amount)
                            progress = ((current * i.current) + (already_done * i.total[0])) * 100 / (i.total[0] * request.amount)
                            print((current * i.current), (already_done * i.total[0]), (i.total[0] * request.amount))
                            #progress = ((i.current * 100) / (i.total[0])) * len(i.interactions) + (i.total[0] * [x for x in sendable_images if isinstance(x, GenericOutput)])/ request.amount
                            asyncio.run_coroutine_threadsafe(
                                coro=interaction.edit_original_message(content=str(round(progress, 2)) + "%",
                                   files=sendable_images), loop=client.loop)
                    limiter = time.time()
        for request in now:
            sendable_images = []
            for idx, image in enumerate(images[request.interaction]):
                if image != None:
                    with io.BytesIO() as imagebn:
                        image.output.save(imagebn, format="JPEG", subsampling=0, quality=90)
                        imagebn.seek(0)
                        sendable_images.append(discord.File(fp=imagebn, filename=str(image.index) + ".jpg"))
            if sendable_images != []:
                if request.negative_prompt != "":
                    asyncio.run_coroutine_threadsafe(coro=request.interaction.edit_original_message(content=str(
                        request.amount) + " images of '" + str(request.prompt) + "' + (negative: '" + str(request.negative_prompt) + "')",
                                                                                                    files=sendable_images),
                                                     loop=client.loop)
                else:
                    asyncio.run_coroutine_threadsafe(coro=request.interaction.edit_original_message(
                        content=str(request.amount) + " images of '" + request.prompt + "'", files=sendable_images),
                        loop=client.loop)
        model_reused = False
        if len(run_queue) > 1:
            if run_queue[1][0].model.path == now[0].model.path:
                run_queue[1][0].model = now[0].model
                model_reused = True
            # simplify this for a moment
            #elif len(run_queue) > 2:
            #    if run_queue[2][0].model.path == now[0].model.path:
            #        run_queue[2][0].model = now[0].model.to('cpu')
            #        model_reused = True
        if not model_reused:
            now[0].model.del_model()
            del now[0].model
        #now[0].model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        run_queue.pop(0)


def model_runner():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_model_runner())


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
            required=False,
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
    if not negative_prompt: negative_prompt = ""
    await interaction.response.send_message("Generation has been queued.")
    print("adding generation")
    # (self, model, prompt, negative_prompt, amount, interaction)
    prompt_queue.append(FactoryRequest(model=model, prompt=prompt, negative_prompt=negative_prompt, amount=images,
                                       interaction=interaction))


threading.Thread(target=model_factory, daemon=True).start()
threading.Thread(target=model_runner, daemon=True).start()
client.run(TOKEN)
