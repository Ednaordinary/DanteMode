# publicmain.py
from gettext import install
import os
import subprocess
import nextcord as discord
from nextcord import message
from dotenv import load_dotenv
from nextcord.ext import commands
from random import *
import numpy as np
import image
import torch
import queue
import time
from big_sleep import Imagine
from PIL import Image as Imageb
import shutil
import random
from deep_daze import Imagine as Imagine2
from deep_daze import Imagine as Imagine2
from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer

print("Loading Access")
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
client = discord.Client()

print("Starting initial worker")
# Thread(target=worker, daemon=True).start() # queue and multithread not currently working, does nothing good
print("Finished starting initial worker")

@client.event
async def on_error(event, *args):
    with open('err.log', 'a') as f:
        if event == 'on_message':
            f.write(f'Unhandled message: {args[0]}\n')
        else:
            raise

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')
    await client.change_presence(
        activity=discord.Activity(type=discord.ActivityType.listening, name=' (your user) shout at me'))


@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(
        f'Hi {member.name}, welcome to this Discord server!'
    )


@client.event
async def on_message(message):
    if 'dante gif,' in message.content.lower():
        if message.author.bot:
            return
        await message.channel.send("Generating Gif.")
        numsave = 0
        dreamprompt = message.content.lower()
        print(dreamprompt)
        removal = ['[', 'dante', 'gif,', '', ']', '/','{','}']
        dreampromptremoval = dreamprompt.split()
        dreampromptconnect = [word for word in dreampromptremoval if word not in removal and word.isalnum()]
        dreampromptfinal = " ".join(map(str, dreampromptconnect))
        print(dreampromptfinal)
        dream = Imagine(text=dreampromptfinal, epochs=20, iterations=20, image_size=256, save_every=1,
                        open_folder=False, num_cutouts=64)
        for a in range(20):
            for b in range(20):
                dream.train_step(a, b)
                numsave += 1
                filename = dreampromptfinal.replace(' ', '_')
                image = Imageb.open(f'./{filename}.png')
                path = f'./Dante/Collection/DanteGif/{filename}/'
                if not os.path.exists(path):
                    os.makedirs(path)
                shutil.copy2(f"./{filename}.png", f"{path}{filename}{numsave}.png")
        await message.channel.send("Generating done, Creating Gif.")
        num = random.sample(range(1, 1000),1)
        filename = dreampromptfinal.replace(' ', '_')
        os.system(f"ffmpeg -y -framerate 20 -i './Dante/Collection/DanteGif/{filename}/{filename}%d.png' -pix_fmt yuv420p './Dante/Collection/DanteGif/{filename}{num}.mp4'")
        # os.system(f"'/storage1/Other/bots/Dante/filereduce.sh' '/storage1/Other/bots/Dante/DanteGif/GifCollection1/{filename}.mp4' 8")
        await message.channel.send(file=discord.File(f'./Dante/Collection/DanteGif/{filename}{num}.mp4'))
        # await message.channel.send(f"Added to queue, estimated time {size * 10} minutes")
        print("Finish")
    if 'dante,' in message.content.lower():
        if message.author.bot:
            return
        await message.channel.send("Generating image.")
        dreamprompt = message.content.lower()
        print(dreamprompt)
        removal = ['[', 'dante,', ']', '/','{','}']
        dreampromptremoval = dreamprompt.split()
        dreampromptconnect = [word for word in dreampromptremoval if word not in removal and word.isalnum()]
        dreampromptfinal = " ".join(map(str, dreampromptconnect))
        print(dreampromptfinal)
        dream = Imagine(text=dreampromptfinal, epochs=20, iterations=20, image_size=256, save_every=1,
                        open_folder=False, num_cutouts=64)
        dream()
        filename = dreampromptfinal.replace(' ', '_')
        path = f'./Dante/Collection/Dante/'
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy2(f"./{filename}.png", f"{path}{filename}.png")
        await message.channel.send("Generating done, Enhancing resolution.")
        # f"enhance --zoom=2 {filename}.png"
        subprocess.call([f"'./enhance.sh' {path}{filename}"], shell=True)
        # await message.channel.send("Upscaling done, compressing down to discord file size (in testing).")
        # subprocess.call([f"'/storage1/Other/bots/Dante/filereduce.sh' {filename}_ne2x_ne2x.png 8"], shell=True)
        # image = Image(f'./{filename}.png')
        await message.channel.send(file=discord.File(f"{path}{filename}_ne4x.png"))

        # await message.channel.send(f"Added to queue, estimated time {size * 10} minutes")
        print("Finish")
    if 'dante alt,' in message.content.lower():
        if message.author.bot:
            return
        await message.channel.send("Generating image.")
        dreamprompt = message.content.lower()
        print(dreamprompt)
        removal = ['[', 'dante', 'alt,', ']', '/','{','}']
        dreampromptremoval = dreamprompt.split()
        dreampromptconnect = [word for word in dreampromptremoval if word not in removal and word.isalnum()]
        dreampromptfinal = " ".join(map(str, dreampromptconnect))
        print(dreampromptfinal)
        dream = Imagine2(
            text=dreampromptfinal,
            save_every=1,
            epochs=20,
            batch_size=32,
            num_layers=48,
            iterations=20,
            image_width=256,
            open_folder=False,
            save_progress=False,
            gradient_accumulate_every=1
        )
        dream()
        # await message.channel.send("Generating done, Enhancing resolution.")
        filename = dreampromptfinal.replace(' ', '_')
        # f"enhance --zoom=2 {filename}.png"
        # subprocess.call([f"ffmpeg -i {filename}.000399.jpg {filename}.000399.png"], shell=True)
        # +subprocess.call([f"'/storage1/Other/bots/Dante/enhance.sh' {filename}.000399"], shell=True)
        # await message.channel.send("Upscaling done, compressing down to discord file size (in testing).")
        # subprocess.call([f"'/storage1/Other/bots/Dante/filereduce.sh' {filename}_ne2x_ne2x.png 8"], shell=True)
        # image = Image(f'./{filename}.png')
        path = f'./Dante/Collection/Dante/'
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy2(f"./{filename}.png", f"{path}{filename}.png")
        await message.channel.send(file=discord.File(f"{path}{filename}.jpg"))

        # await message.channel.send(f"Added to queue, estimated time {size * 10} minutes")
        print("Finish")
    if 'happy birthday' in message.content.lower():
        if message.author.bot:
            return
        await message.channel.send("Happy Birthday! 🎈🎉")


client.run(TOKEN)
