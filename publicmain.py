import asyncio
import io
import os
import re
import subprocess
import sys
import time
from typing import Optional

import nextcord as discord
from dotenv import load_dotenv
import threading
from collections import deque
from PIL import Image

#Variables for deployment
modelpath = "/path/to/models" #Path for models
internalerrorchannel = 12345 #ID for channel to send error messages to for debugging

print("Loading Access")
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)
queue = deque([])

class GenerateModal(discord.ui.Modal):
    def __init__(self):
        super().__init__(
            title="Generate Image",
            timeout=10 * 60,  # 10 minutes
        )

        self.prompt = discord.ui.TextInput(
            label="Prompt",
            style=discord.TextInputStyle.paragraph,
            min_length=1,
            max_length=255,
        )
        self.add_item(self.prompt)
        
        self.model = discord.ui.TextInput(
            label="Model (Must be one of: normal)",
            min_length=2,
            max_length=6,
            default_value="normal",
        )
        self.add_item(self.model)
        
        self.images = discord.ui.TextInput(
            label="Images (Must be a number between 1-10)",
            min_length=1,
            max_length=2,
            default_value="2",
        )
        self.add_item(self.images)

    async def callback(self, interaction: discord.Interaction) -> None:
        valid = True
        try:
            images = int(self.images.value)
        except Exception:
            response = "Please enter a number for images"
            valid = False
        if valid == True:
            if images > 10 or images < 1:
                response = "Please enter a number between 1 and 10 for images"
                valid = False
        if valid == True:
            model = str(self.model.value).lower()
            if model == "normal":
                model = "sdiff"
            else:
                response = "Please enter one of the listed models"
                valid = False
        if valid == True:
            prompt = self.prompt.value
            regex = re.compile('[^a-zA-Z0-9 ]')
            prompt = regex.sub('', prompt)
            if not model in ["put video models here"]:
                response = f"Generating image with the prompt {prompt}, {images} output images, and {model}% model."
                queue.append([prompt, model, images, interaction, "text2img"])
            else:
                response = f"Generating video with the prompt {prompt}, {images} output videos, and {model}% model."
                queue.append([prompt, model, images, interaction, "text2vid"])
        await interaction.send(response)

# Currently unused
class EditModal(discord.ui.Modal):
    def __init__(self):
        super().__init__(
            title="Edit Image",
            timeout=10 * 60,  # 10 minutes
        )

        self.prompt = discord.ui.TextInput(
            label="Prompt",
            style=discord.TextInputStyle.paragraph,
            min_length=0,
            max_length=255,
        )
        self.add_item(self.prompt)

        self.images = discord.ui.TextInput(
            label="Images (Must be a number between 1-5)",
            min_length=1,
            max_length=1,
            default_value="2",
        )
        self.add_item(self.images)

        self.strength = discord.ui.TextInput(
            label="Strength (Must be a percent between 1-100%)",
            min_length=1,
            max_length=10,
            default_value="50%",
        )
        self.add_item(self.strength)

    async def callback(self, interaction: discord.Interaction) -> None:
        valid = True
        try:
            images = int(self.images.value)
        except Exception:
            response = "Please enter a number for images"
            valid = False
        if valid == True:
            if images > 5 or images < 1:
                response = "Please enter a number between 1 and 5 for images"
                valid = False
        if valid == True:
            try:
                strength = float(self.strength.value.replace("%", "")) / 100
            except Exception:
                response = "Please enter a percentage for strength"
                valid = False
        if valid == True:
            if strength > 1 or strength < 0:
                response = "Please enter a percentage between 1 and 100% for strength"
                valid = False
        if valid == True:
            response = f"Editing image with the prompt {self.prompt.value}, {images} output images, and {strength * 100}% strength."
        await interaction.send(response)

class UpscaleOnly(discord.ui.View):
    def __init__(self, *, timeout=60 * 60):
        super().__init__(timeout=timeout)
    @discord.ui.button(label="Upscale", style=discord.ButtonStyle.primary)
    async def upscale_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        button.style = discord.ButtonStyle.secondary
        button.label = "Upscaling..."
        await interaction.response.edit_message(view=self)
        num = 0
        scalemodel = "esrgan"
        path = str(str(modelpath) + "/upscale/" + scalemodel + "/")
        upscalemessage = await interaction.channel.send("Your upscaled images will appear here")
        for attachment in interaction.message.attachments:
            await attachment.save(fp=path + str(num) + ".jpg")
            subprocess.run([path + "scale.sh " + path + str(num) + ".jpg " + path + str(num) + "_scale.jpg"],
                shell=True, cwd=path)
            subprocess.run([
                "ffmpeg -y -i " + path + str(num) + "_scale.jpg " + path + str(num) + "_scale2.jpg"],
                shell=True)
            num += 1
            for i in range(num):
                files = []
                for i in range(num):
                    files.append(discord.File(path + str(i) + "_scale2.jpg"))
            await upscalemessage.edit("Here is your upscaled image" if num == 1 else "Here are your upscaled images", files=files)
        button.style = discord.ButtonStyle.green
        button.label = "Upscaling Complete!"
        await interaction.message.edit(view=self)

async def interactioneditmessage(interaction, content, view=None, files=None):
    if view == "UpscaleOnly": view = UpscaleOnly()
    await interaction.edit_original_message(content=content, view=view, files=files)

async def interactioneditmessagewithcancel(interaction, content, gen=None, files=None):
    if gen is not None:
        view = CancelGen(gen)
    else:
        view = None
    await interaction.edit_original_message(content=content, view=view, files=files)

def updategenstatus():
    nonmutatablequeue = queue
    for generation in nonmutatablequeue:
        time.sleep(0.1)
        pos = nonmutatablequeue.index(generation) + 1
        interaction = generation[3]
        asyncio.run_coroutine_threadsafe(coro=interactioneditmessagewithcancel(interaction, "Prompt submitted to queue.\nThis message will be edited when it is done.\nYour position in the queue: " + str(pos) + "/" + str(len(nonmutatablequeue))), loop=client.loop)

def watcher():
    while True:
        time.sleep(0.1)
        if len(queue) > 0:
            run = queue[0]
            prompt = run[0]
            model = run[1]
            images = run[2]
            interaction = run[3]
            type = run[4]
            queue.popleft()
            threading.Thread(target=updategenstatus, daemon=True).start()
            if type == "text2img":
                if os.path.exists(str(modelpath) + "/text2img/" + str(model) + "/gen.py"):
                    asyncio.run_coroutine_threadsafe(coro=interactioneditmessage(interaction, "Generating now..."), loop=client.loop)
                    timestart = time.perf_counter()
                    subprocess.run(modelpath + "/text2img/" + model + "/venv/bin/python " + modelpath + "/text2img/" + model + "/gen.py '" + prompt + "' " + str(images), shell=True, cwd = modelpath + "/text2img/" + model)
                    files = []
                    for i in range(images):
                        files.append(discord.File(str(str(modelpath) + "/text2img/" + str(model) + "/" + str(i) + ".jpg")))
                    view = "UpscaleOnly"
                    asyncio.run_coroutine_threadsafe(coro=interactioneditmessage(interaction=interaction, content=str(images) + " images of '" + prompt + "' using the " + model + " model, generated in " + str(round(time.perf_counter() - timestart)) + " seconds:", files=files, view=view), loop=client.loop)
                else:
                    asyncio.run_coroutine_threadsafe(coro=interactioneditmessage(interaction, "Internal Error: Model '" + str(model) + "' not found"), loop=client.loop)
            elif type == "text2vid":
                if os.path.exists(str(modelpath) + "/text2vid/" + str(model) + "/gen.py"):
                    asyncio.run_coroutine_threadsafe(coro=interactioneditmessage(interaction, "Generating now..."), loop=client.loop)
                    timestart = time.perf_counter()
                    subprocess.run(modelpath + "/text2vid/" + model + "/venv/bin/python " + modelpath + "/text2vid/" + model + "/gen.py '" + prompt + "' " + str(images), shell=True, cwd = modelpath + "/text2vid/" + model)
                    files = []
                    for i in range(images):
                        files.append(discord.File(str(str(modelpath) + "/text2vid/" + str(model) + "/" + str(i) + ".mp4")))
                    asyncio.run_coroutine_threadsafe(coro=interactioneditmessage(interaction=interaction, content=str(images) + " videos of '" + prompt + "' using the " + model + " model, generated in " + str(round(time.perf_counter() - timestart)) + " seconds:", files=files), loop=client.loop)
                else:
                    asyncio.run_coroutine_threadsafe(coro=interactioneditmessage(interaction, "Internal Error: Model '" + str(model) + "' not found"), loop=client.loop)
            else:
                asyncio.run_coroutine_threadsafe(coro=interactioneditmessage(interaction, "Internal Error: Type '" + str(type) + "' not found"), loop=client.loop)


@client.event
async def on_error(event, *args, **kwargs):
    print(sys.exc_info())
    raise
    embed = discord.Embed(title="An error occured!", color=discord.Colour.from_rgb(253, 253, 253),
                          description=str(sys.exc_info()))
    await client.get_channel(internalerrorchannel).send(embed=embed)
    embed = discord.Embed(color=discord.Colour.from_rgb(253, 253, 253),
                          description=str(args))
    await client.get_channel(internalerrorchannel).send(embed=embed)

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')

@client.slash_command(description="Generates an image from the prompt")
async def generate(
        interaction: discord.Interaction,
        prompt: str = discord.SlashOption(
            name="prompt",
            required=True,
            description="The prompt to generate the image off of",
            max_length=255,
        ),
        model: Optional[str] = discord.SlashOption(
            name="model",
            choices={"normal": "sdiff"},
            required=False,
            description="The model to use to generate the image",
        ),
        images: Optional[int] = discord.SlashOption(
            name="images",
            choices={"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10},
            required=False,
            description="How many images to generate (more will take longer)"
        )
):
    regex = re.compile('[^a-zA-Z0-9 ]')
    prompt = regex.sub('', prompt)
    if model is None: model = "sdiff"
    if images is None: images = 2
    if not model in ["put video models here"]:
        queue.append([prompt, model, images, interaction, "text2img"])
    else:
        queue.append([prompt, model, images, interaction, "text2vid"])
    await interaction.send("Submitting to queue...")
    threading.Thread(target=updategenstatus, daemon=True).start()

@client.slash_command(description="Sends a form to generate an image from the prompt")
async def generate_form(interaction: discord.Interaction):
    modal = GenerateModal()
    await interaction.response.send_modal(modal)

threading.Thread(target=watcher, daemon=True).start()
client.run(TOKEN)
