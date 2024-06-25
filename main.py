from diffusers import AutoencoderTiny
from models import GenericModel, OptimizedModel, IntermediateModel, GenericOutput, IntermediateOutput, RunStatus
from dotenv import load_dotenv
import nextcord as discord
import threading
import torch
import time
import gc

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
        self.negative_prompts = negative_prompts
        self.interactions = interactions

def model_factory():
    global prompt_queue
    global run_queue
    while True
        while prompt_queue == []:
            time.sleep(0.01)
        while len(run_queue) > 1:
            time.sleep(0.01)
        for idx, prompt in enumerate(prompt_queue[0]): # prompt, negative_prompt, interaction, index
            if idx == 0:
                current_model = prompt.model
                flattened_run = RunRequest(model=model_translations[prompt.model], prompts=[prompt.prompt]*prompt.amount, negative_prompts=[prompt.negative_prompt]*prompt.amount, interactions=[prompt.interactions]*prompt.amount)
            else:
                prompts = flattened_run.prompts + [prompt.prompt]*prompt.amount
                negative_prompts = flattened_run.negative_prompts + [prompt.negative_prompt]*prompt.amount
                interactions = flattened_run.interactions + [prompt.negative_prompt]*prompt.amount
                flattened_run = RunRequest(model=flattened_run.model, prompts=prompts, negative_prompts = negative_prompts, interactions=interactions)
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

def model_runner():
    global run_queue
    while True:
        while 

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
        negative_prompt: str = discord.SlashOption(
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
    interaction.response.send_message("Generation has been queued.")
    #(self, model, prompt, negative_prompt, amount, interaction)
    prompt_queue.append(FactoryRequest(model=model, prompt=prompt, negative_prompt=prompt, amount=images, interaction=interaction))

threading.Thread(target=model_factory, daemon=True).start()
client.run(TOKEN)
