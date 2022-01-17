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

print("Creating Models")
has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
print("Creating base model...")
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '50' # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))
print("Creating upsampler model...")
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
def show_images(batch: th.Tensor, filename):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    theimage = Imageb.fromarray(reshaped.numpy())
    theimage.save(f"/storage1/Other/bots/Dante/Collection 12/{filename}.png")
print("Creating CLIP")
clip_model = create_clip_model(device=device)
clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', device))
clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', device))
print("Finished creating models")

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
        if 'dante, help' in message.content.lower():
            await message.channel.send("I am Dante, created by Ednaordinary#6602. I have several commands listed below:")
            await message.channel.send("'Dante, (prompt)' This is my most basic command, which generates an image based on the prompt. (45 seconds)")
            # await message.channel.send("'Dante zoom, (prompt)' This command is similar to dante gif, except it generates a zooming and rotating video. This uses a separate model due to compatibility issues. (5-7 minutes)")
            await message.channel.send("'Dante gif, (prompt)' This command generates a gif of the prompt, by recording each iteration. This uses a separate model due to compatibility issues. (5-7 minutes)")
            await message.channel.send("'Dante alt, (prompt)' This command generates an image based on the prompt using an alternative model.")
            # await message.channel.send("'Dante aph, (prompt)' This command uses an alternative aph model, with some very interesting results. (5 minutes)")
            # await message.channel.send("'Dante aph zoom, (prompt)' This command uses an alternative aph model while also zooming in, with some very interesting results. (5 minutes)")
            # await message.channel.send("'Dante aph fast, (prompt)' This command uses an alternative aph model along with fewer iterations, but lesser quality. (30 seconds)")
            # await message.channel.send("'Dante aph zoom fast, (prompt)' This command uses an alternative aph model along with fewer iterations while also zooming in. (30 seconds)")
            # await message.channel.send("'Dante old, (prompt)' This command uses the old dante model. (10 minutes)")
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
