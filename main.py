import nextcord as discord
from dotenv import load_dotenv
import threading
import torch
import gc
from models import OptimizedModel

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)
prompt_queue = []
run_queue = []
model_translations = {
    "sd": OptimizedModel("runwayml/stable-diffusion-v1-5", "image", 10, 25),
}

class FactoryRequest:
    def __init__(self, model, prompt, negative_prompt, amount):
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.amount = amount

class RunRequest:
    def __init__(self, model, prompts, negative_prompts):
        self.model = model
        self.prompts = prompts
        self.negative_prompts = negative_prompts

def model_factory():
    global prompt_queue
    global run_queue
    while True
        if prompt_queue == []:
            time.sleep(0.01)
        for idx, prompt in enumerate(prompt_queue): # prompt, negative_prompt, interaction, index
            if idx == 0:
                current_model = prompt.model
                flattened_run = RunRequest(model=model_translations[prompt.model], prompts=[x ])
            else:
                
            #asyncio.run_coroutine_threadsafe(coro=prompt.interaction.message.edit(content="Model loading..."), loop=client.loop)

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')

threading.Thread(target=model_factory, daemon=True).start()
client.run(TOKEN)
