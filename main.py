from diffusers import AutoencoderTiny
from models import GenericModel, OptimizedModel, IntermediateModel, GenericOutput, IntermediateOutput, RunStatus, Prompt
from dotenv import load_dotenv
import nextcord as discord
import threading
import asyncio
import torch
import time
import gc
import io

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)
prompt_queue = []
run_queue = []
model_translations = {
    (self, path, out_type, max_latent, steps, mini_vae)
    "sd": IntermediateModel(path="runwayml/stable-diffusion-v1-5", out_type="image", max_latent=10, steps=25, mini_vae=AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)),
}
default_images = {
    "sd": 10
}
interactions = {}

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

class Output:
    def __init__(self, output, out_type):
        self.output = output
        self.out_type = out_type

def model_factory():
    global prompt_queue
    global run_queue
    while True
        while prompt_queue == []:
            time.sleep(0.01)
        while len(run_queue) > 1:
            time.sleep(0.01)
        for idx, prompt in enumerate(prompt_queue[0]): # self, model, prompt, negative_prompt, amount, interaction
            if idx == 0:
                current_model = prompt.model
                flattened_run = RunRequest(model=model_translations[prompt.model], prompts=prompt)
            else:
                flattened_run = RunRequest(model=flattened_run.model, prompts=flattened_run.prompts + prompt)
        last_interaction = None
        for interaction in flattened_run.interactions:
            if interaction != last_interaction:
                asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message("Model loading...", loop=client.loop)
            last_interaction = interaction
        flattened_run.model.to("cpu")
        if type(flattened_run.model) = IntermediateModel:
            flattened_run.mini_vae.to("cpu")
        for interaction in flattened_run.interactions:
            if interaction != last_interaction:
                asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message("Model loaded to cpu", loop=client.loop)
            last_interaction = interaction
        run_queue.append(flattened_run)
        prompt_queue.pop(0)

async def async_model_runner():
    global run_queue
    while True:
        if run_queue == []:
            time.sleep(0.01)
        now = run_queue[0] # this is a list of FactoryRequests. self, model, prompt, negative_prompt, amount, interaction
        prompts = []
        images = {}
        #this is so much simpler why didn't I think of it earlier??
        now.model.to('cuda')
        for request in now:
            for i in range(request.amount):
                prompts.append(Prompt(prompt=request.prompt, request=request.negative_prompt, interaction=request.interaction, index=i))
                images[request.interaction] = [None]*request.amount
                asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message("Model loaded to gpu", loop=client.loop)
        #prompts = []
        #index_amts = []
        #last_interaction = None
        #for idx, interaction in enumerate(now.interactions):
        #    if interaction == last_interaction:
        #        index_amts[-1] += 1
        #    else:
        #        index_amts.append(1)
        #        last_interaction = last_interaction
        #last_interaction = None:
        #real_idx = None
        #for idx, interaction in enumerate(now.interactions): # (self, model, prompts, negative_prompts, interactions)
        #    if interaction == last_interaction:
        #        real_idx += 1
        #    else:
        #        real_idx 
        #    # (self, prompt, negative_prompt, interaction, index):
        #    prompts.append(Prompt(prompt=request.prompt, negative_prompt=now.negative_prompts[idx], now.interactions[idx], idx))
        limiter = time.time()
        async for i in now.model.call(prompts):
            if type(i) == GenericOutput: #(self, output, out_type, interaction, index)
                #This event is final, meaning this image is done.
                images[i.interaction][index] = Output(output=i.output, out_type=i.out_type)
            if type(i) == IntermediateOutput:
                
            if type(i) == RunStatus:
                

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
            name="negative prompt",
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
    if not negative_prompt: negative_prompt=None
    interaction.response.send_message("Generation has been queued.")
    #(self, model, prompt, negative_prompt, amount, interaction)
    prompt_queue.append(FactoryRequest(model=model, prompt=prompt, negative_prompt=negative_prompt, amount=images, interaction=interaction))

threading.Thread(target=model_factory, daemon=True).start()
client.run(TOKEN)
