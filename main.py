import random
import subprocess
import sys

import torchaudio

from models.audio import SAUDIOModel
from models.generic import GenericModel, GenericOutput, RunStatus, Prompt, FinalOutput
from models.intermediate import IntermediateOutput, IntermediateOptimizedModel, IntermediateModel
from models.pasi import PASIModel
from models.sd import SDXLModel, SDXLTModel, SD3Model, SCASCModel, SDXLDSModel, SDXLJXModel, SDDSModel
from models.optimized import OptimizedModel
from diffusers.utils import numpy_to_pil, export_to_video
from dotenv import load_dotenv
from typing import Optional
import nextcord as discord
from PIL import Image
import threading
import asyncio
import torch
import time
import PIL
import gc
import io
import os

from models.upscale import LDMUpscaleModel
from models.video import ZSVideoModel, SVDVideoModel

torch.backends.cuda.matmul.allow_tf32 = True

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)
prompt_queue = []
run_queue = None
current_model_path = None
model_translations = {
    "sd": IntermediateOptimizedModel(path="runwayml/stable-diffusion-v1-5", out_type="image", max_latent=50, steps=30,
                                     mini_vae="madebyollin/taesd"),
    "sd2": IntermediateOptimizedModel(path="stabilityai/stable-diffusion-2-1", out_type="image", max_latent=30,
                                      steps=30,
                                      mini_vae="madebyollin/taesd"),
    "sdxl": SDXLModel(path="stabilityai/stable-diffusion-xl-base-1.0", out_type="image", max_latent=15, steps=35,
                      mini_vae="madebyollin/taesdxl"),
    "sdxl-ds": SDXLDSModel(path="Lykon/dreamshaper-xl-1-0", out_type="image", max_latent=15, steps=35,
                           mini_vae="madebyollin/taesdxl"),
    "sdxl-jx": SDXLJXModel(path="RunDiffusion/Juggernaut-X-v10", out_type="image", max_latent=15, steps=35,
                           mini_vae="madebyollin/taesdxl"),
    "sdxl-t": SDXLTModel(path="stabilityai/sdxl-turbo", out_type="image", max_latent=100, steps=2),
    "sd-ds": SDDSModel(path="Lykon/dreamshaper-8", out_type="image", max_latent=50, steps=30,
                       mini_vae="madebyollin/taesd"),
    "sd3-m": SD3Model(path="stabilityai/stable-diffusion-3-medium-diffusers", out_type="image", max_latent=10, steps=35,
                      mini_vae="madebyollin/taesd3"),
    "scasc": SCASCModel(path="stabilityai/stable-cascade", out_type="image", max_latent=10, steps=20),
    "pa-si": PASIModel(path="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", out_type="image", max_latent=20, steps=35,
                       mini_vae="madebyollin/taesdxl"),
    "s-video": SVDVideoModel(path="stabilityai/stable-video-diffusion-img2vid-xt-1-1", out_type="video-zs",
                             max_latent=1, steps=35, mini_vae="madebyollin/taesdxl"),
    "zs-video": ZSVideoModel(path="cerspense/zeroscope_v2_576w", out_type="video-zs", max_latent=1, steps=40),
    "s-audio": SAUDIOModel(path="stabilityai/stable-audio-open-1.0", out_type="s-audio", max_latent=5, steps=100),
}
default_images = {
    "sd": 10,
    "sd2": 10,
    "sdxl": 10,
    "sdxl-ds": 10,
    "sdxl-jx": 10,
    "sdxl-t": 10,
    "sd-ds": 10,
    "sd3-m": 5,
    "scasc": 10,
    "pa-si": 10,
    "s-video": 1,
    "zs-video": 3,
    "s-audio": 3,
}
images = {}


async def edit_any_message(message, content, files, view, request):
    if view == "AgainAndUpscale":
        view = AgainAndUpscaleButton(request=request)
    for i in range(5):  # Sometimes we'll get mac address errors due to load balancing
        try:
            params = {"content": content, "files": files, "view": view}
            params = {k: v for k, v in params.items() if v is not None}
            if isinstance(message, discord.Interaction):
                await message.edit_original_message(**params)
            else:
                await message.edit(**params)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(repr(e))
            pass
        else:
            return


class AgainButton(discord.ui.View):
    def __init__(self, *, timeout=None, request):
        super().__init__(timeout=timeout)
        self.request = request

    @discord.ui.button(label="Again", style=discord.ButtonStyle.primary)
    async def again_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        message = await interaction.channel.send("Generation has been queued.", view=self)
        global prompt_queue
        prompt_queue.append(
            FactoryRequest(model=self.request.model, prompt=self.request.prompt,
                           negative_prompt=self.request.negative_prompt,
                           amount=self.request.amount,
                           interaction=message))
        button.style = discord.ButtonStyle.secondary
        await interaction.response.edit_message(view=self)


class AgainAndUpscaleButton(discord.ui.View):
    def __init__(self, *, timeout=None, request):
        super().__init__(timeout=timeout)
        self.request = request

    @discord.ui.button(label="Again", style=discord.ButtonStyle.primary)
    async def again_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        message = await interaction.channel.send("Generation has been queued.", view=self)
        global prompt_queue
        prompt_queue.append(
            FactoryRequest(model=self.request.model, prompt=self.request.prompt,
                           negative_prompt=self.request.negative_prompt,
                           amount=self.request.amount,
                           interaction=message))
        button.style = discord.ButtonStyle.secondary
        await interaction.response.edit_message(view=self)

    @discord.ui.button(label="Upscale", style=discord.ButtonStyle.primary)
    async def upscale_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        message = await interaction.channel.send("Upscale has been queued.")
        global prompt_queue
        images = []
        button.style = discord.ButtonStyle.secondary
        await interaction.response.edit_message(view=self)
        for attachment in interaction.message.attachments:
            image = await attachment.read()
            images.append(Image.open(io.BytesIO(image)).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS))
        prompt_queue.append(FactoryRequest(
            model=LDMUpscaleModel(path="CompVis/ldm-super-resolution-4x-openimages", out_type="image", max_latent=1,
                                  steps=40), prompt=images, negative_prompt="",
            amount=len(interaction.message.attachments),
            interaction=message))


class FactoryRequest:
    def __init__(self, model, prompt, negative_prompt, amount, interaction):
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.amount = amount
        self.interaction = interaction


def model_factory():
    global prompt_queue
    global run_queue
    global current_model_path
    global live_sessions
    while True:
        if prompt_queue != [] and run_queue != None:
            if prompt_queue[0].model.path == run_queue[0].model.path:
                run_queue.append(prompt_queue[0])
                prompt_queue.pop(0)
        if prompt_queue != [] and run_queue == None:  # has to be reevaluated
            device = 'gpu'
            if not prompt_queue[0].model.path == current_model_path:
                print("loading model to cpu")
                prompt_queue[0].model.to('cpu')
                device = 'cpu'
            tmp_queue = []
            tmp_path = prompt_queue[0].model.path
            pop_amt = 0
            for prompt in prompt_queue:
                if not prompt.model.path == tmp_path:
                    break
                tmp_queue.append(prompt)
                send_load_message = True
                for u, int in live_sessions.items():
                    if prompt.interaction == int:
                        send_load_message = False
                        break
                if send_load_message:
                    asyncio.run_coroutine_threadsafe(
                        coro=edit_any_message(prompt.interaction, "Model loaded to " + device, None, None, None),
                        loop=client.loop
                    )
                pop_amt += 1
            for i in range(pop_amt): prompt_queue.pop(0)
            run_queue = tmp_queue
            del tmp_queue
            del tmp_path
            gc.collect()
        time.sleep(0.01)


async def async_model_runner():
    global run_queue
    global images
    global current_model_path
    finalized = {}
    updated = {}
    while True:
        while not run_queue:
            time.sleep(0.01)
        model_passthrough = True
        now = run_queue
        run_queue = None
        current_model_path = now[0].model.path
        send_cuda_message = False
        try:
            now[0].model.model.device
        except:
            now[0].model.to('cuda')
            send_cuda_message = True
        else:
            if now[0].model.model.device.type != "cuda":
                now[0].model.to("cuda")
                send_cuda_message = True
        start_time = time.time()
        prompts = []
        for request in now:
            if isinstance(now[0].model, LDMUpscaleModel):
                for idx, i in enumerate(request.prompt):
                    prompts.append(Prompt(prompt=i, negative_prompt=request.negative_prompt,
                                          interaction=request.interaction, index=idx, parent_amount=request.amount))
            else:
                for i in range(request.amount):
                    prompts.append(Prompt(prompt=request.prompt, negative_prompt=request.negative_prompt,
                                          interaction=request.interaction, index=i, parent_amount=request.amount))
            images[request.interaction] = [None] * request.amount
            updated[request.interaction] = False
            finalized[request.interaction] = False
            if send_cuda_message:
                asyncio.run_coroutine_threadsafe(
                    coro=edit_any_message(request.interaction, "Model loaded to gpu", None, None, None),
                    loop=client.loop)
        limiter = time.time()
        with torch.no_grad():
            try:
                async for i in now[0].model.call(prompts):
                    if isinstance(i, FinalOutput):
                        for output in i.outputs:
                            images[output.prompt.interaction][output.prompt.index] = output
                            updated[output.prompt.interaction] = False
                        for interaction in list(set([x.prompt.interaction for x in i.outputs])):
                            if True:
                                for prompt in now:
                                    if prompt.interaction == interaction:
                                        this_request = prompt
                                        break
                                sendable_images = [None] * this_request.amount
                                for_decoding = []
                                for image in images[interaction]:
                                    if image != None:
                                        if isinstance(image.output, PIL.Image.Image):
                                            imagebn = io.BytesIO()
                                            image.output.save(imagebn, format='JPEG', quality=80)
                                            imagebn.seek(0)
                                            sendable_images[image.prompt.index] = discord.File(fp=imagebn, filename=str(
                                                image.prompt.index) + ".jpg")
                                        elif image.out_type[0] == "video-zs":
                                            # unfortunately, we have to make a temporary file
                                            video_path = str(random.randint(1, 10000000)) + ".mp4"
                                            export_to_video(image.output, video_path)
                                            subprocess.check_call(
                                                "ffmpeg -i " + str(video_path) + " redo-" + str(video_path), shell=True)
                                            with open("redo-" + video_path, "rb") as video_file:
                                                videobn = io.BytesIO(video_file.read())
                                            videobn.seek(0)
                                            sendable_images[image.prompt.index] = discord.File(fp=videobn, filename=str(
                                                image.prompt.index) + ".mp4")
                                            os.remove(video_path)
                                            os.remove("redo-" + video_path)
                                        elif image.out_type[0] == "s-audio":
                                            audio_path = str(random.randint(1, 10000000))
                                            torchaudio.save(audio_path + ".wav", image.output, 44100)
                                            subprocess.check_call('ffmpeg -y -f lavfi -i "color=c=0x' + str(
                                                os.urandom(12).hex()[
                                                :6]) + ':size=512x512" -i ' + audio_path + '.wav -r 1 -c:v libx264 -crf 50 -b:a 72k  -t 45 ' + audio_path + ".mp4",
                                                                  shell=True)
                                            with open(audio_path + ".mp4", "rb") as audio_file:
                                                audiobn = io.BytesIO(audio_file.read())
                                            sendable_images[image.prompt.index] = discord.File(fp=audiobn, filename=str(
                                                image.prompt.index) + ".mp4")
                                            os.remove(audio_path + ".wav")
                                            os.remove(audio_path + ".mp4")
                                        elif image.out_type[0] == "latent-image":
                                            for_decoding.append(image)
                                if for_decoding != None:
                                    for image in for_decoding:
                                        tmp_image = now[0].model.mini_vae.decode(image.output.unsqueeze(0)).sample[0]
                                        tmp_image = tmp_image.to('cpu', non_blocking=False)
                                        gc.collect()
                                        torch.cuda.empty_cache()
                                        tmp_image = numpy_to_pil((tmp_image / 2 + 0.5).permute(1, 2, 0).numpy())[0]
                                        imagebn = io.BytesIO()
                                        tmp_image.thumbnail((256, 256), Image.Resampling.LANCZOS)
                                        tmp_image.save(imagebn, format='JPEG', quality=80)
                                        imagebn.seek(0)
                                        if images[interaction][image.prompt.index] == image:
                                            images[interaction][image.prompt.index].output = tmp_image
                                        gc.collect()
                                        torch.cuda.empty_cache()
                                        sendable_images[image.prompt.index] = discord.File(fp=imagebn, filename=str(
                                            image.prompt.index) + ".jpg")
                                sendable_images = [x for x in sendable_images if x != None]
                                output_count = 0
                                for image in images[interaction]:
                                    if isinstance(image, GenericOutput):
                                        output_count += 1
                                if output_count == len(images[interaction]):
                                    finalized[interaction] = True
                                prompt = images[interaction][0].prompt
                                if finalized[interaction]:
                                    if prompt.negative_prompt != "":
                                        send_message = str(len(sendable_images)) + " images of '" + str(
                                            prompt.prompt) + "' (negative: '" + str(
                                            prompt.negative_prompt) + "') in " + str(
                                            round(time.time() - start_time, 2)) + "s"
                                        send_message = send_message[:-2000]
                                    else:
                                        send_message = str(len(sendable_images)) + " images of '" + str(
                                            prompt.prompt) + "' in " + str(round(time.time() - start_time, 2)) + "s"
                                else:
                                    send_message = None
                                if isinstance(now[0].model, ZSVideoModel) or isinstance(now[0].model,
                                                                                        SAUDIOModel) or isinstance(
                                    now[0].model, SVDVideoModel):
                                    view_type = None
                                else:
                                    if finalized[interaction]:
                                        view_type = "AgainAndUpscale"
                                if isinstance(now[0].model, LDMUpscaleModel):
                                    asyncio.run_coroutine_threadsafe(coro=edit_any_message(interaction, str(len(
                                        sendable_images)) + " images upscaled in " + str(
                                        round(time.time() - start_time, 2)) + "s", [
                                                                                               discord.File(fp=x,
                                                                                                            filename=str(
                                                                                                                idx) + ".jpg")
                                                                                               for idx, x in
                                                                                               enumerate(
                                                                                                   sendable_images)],
                                                                                           None, None),
                                                                     loop=client.loop)
                                else:
                                    asyncio.run_coroutine_threadsafe(
                                        coro=edit_any_message(interaction, send_message, sendable_images, view_type,
                                                              this_request), loop=client.loop)
                                del sendable_images
                    if isinstance(i, IntermediateOutput):
                        images[i.prompt.interaction][i.prompt.index] = i
                        updated[i.prompt.interaction] = True
                    if isinstance(i, RunStatus):
                        if limiter + 2.5 < time.time():
                            limiter = time.time()
                            for interaction in list(set(i.interactions)):
                                if not finalized[interaction]:
                                    if updated[interaction]:
                                        updated[interaction] = False
                                        for prompt in now:
                                            if prompt.interaction == interaction:
                                                this_request = prompt
                                                break
                                        sendable_images = [None] * this_request.amount
                                        for_decoding = []
                                        for image in images[interaction]:
                                            if image != None:
                                                if isinstance(image.output, PIL.Image.Image):
                                                    imagebn = io.BytesIO()
                                                    image.output.save(imagebn, format='JPEG', quality=80)
                                                    imagebn.seek(0)
                                                    sendable_images[image.prompt.index] = discord.File(fp=imagebn,
                                                                                                       filename=str(
                                                                                                           image.prompt.index) + ".jpg")
                                                elif image.out_type[0] == "video-zs":
                                                    # unfortunately, we have to make a temporary file
                                                    # I kinda hate this method, but it's the only way I found
                                                    video_path = str(random.randint(1, 10000000)) + ".mp4"
                                                    export_to_video(image.output, video_path)
                                                    # export_to_video exports a discord unplayable video, must reencode
                                                    subprocess.check_call(
                                                        "ffmpeg -i " + str(video_path) + " redo-" + str(video_path),
                                                        shell=True)
                                                    with open("redo-" + video_path, "rb") as video_file:
                                                        videobn = io.BytesIO(video_file.read())
                                                    videobn.seek(0)
                                                    sendable_images[image.prompt.index] = discord.File(fp=videobn,
                                                                                                       filename=str(
                                                                                                           image.prompt.index) + ".mp4")
                                                    os.remove(video_path)
                                                    os.remove("redo-" + video_path)
                                                elif image.out_type[0] == "s-audio":
                                                    audio_path = str(random.randint(1, 10000000))
                                                    torchaudio.save(audio_path + ".wav", image.output, 44100)
                                                    subprocess.check_call('ffmpeg -y -f lavfi -i "color=c=0x' + str(
                                                        os.urandom(12).hex()[
                                                        :6]) + ':size=512x512" -i ' + audio_path + '.wav -r 1 -c:v libx264 -crf 50 -b:a 72k  -t 45 ' + audio_path + ".mp4",
                                                                          shell=True)
                                                    with open(audio_path + ".wav", "rb") as audio_file:
                                                        audiobn = io.BytesIO(audio_file.read())
                                                    sendable_images[image.prompt.index] = discord.File(fp=audiobn,
                                                                                                       filename=str(
                                                                                                           image.prompt.index) + ".mp4")
                                                    os.remove(audio_path + ".wav")
                                                    os.remove(audio_path + ".mp4")
                                                elif image.out_type[0] == "latent-image":
                                                    for_decoding.append(image)
                                        if for_decoding != None:
                                            for image in for_decoding:
                                                tmp_image = \
                                                now[0].model.mini_vae.decode(image.output.unsqueeze(0)).sample[
                                                    0]
                                                tmp_image = tmp_image.to('cpu', non_blocking=False)
                                                gc.collect()
                                                torch.cuda.empty_cache()
                                                tmp_image = \
                                                numpy_to_pil((tmp_image / 2 + 0.5).permute(1, 2, 0).numpy())[0]
                                                imagebn = io.BytesIO()
                                                tmp_image.thumbnail((256, 256), Image.Resampling.LANCZOS)
                                                tmp_image.save(imagebn, format='JPEG', quality=80)
                                                imagebn.seek(0)
                                                if images[interaction][image.prompt.index] == image:
                                                    images[interaction][image.prompt.index].output = tmp_image
                                                gc.collect()
                                                torch.cuda.empty_cache()
                                                sendable_images[image.prompt.index] = discord.File(fp=imagebn,
                                                                                                   filename=str(
                                                                                                       image.prompt.index) + ".jpg")
                                        sendable_images = [x for x in sendable_images if x != None]
                                        output_count = 0
                                        for image in images[interaction]:
                                            if isinstance(image, GenericOutput) and not isinstance(image,
                                                                                                   IntermediateOutput):
                                                output_count += 1
                                        if output_count == len(images[interaction]):
                                            finalized[interaction] = True
                                        current = 0
                                        for x in i.interactions:
                                            if x == interaction:
                                                current += 1
                                        progress = ((current * i.current) + (output_count * i.total[0])) * 100 / (
                                                i.total[0] * this_request.amount)
                                        send_message = str(round(progress, 2)) + "% " + str(
                                            round(time.time() - start_time, 2)) + "s"
                                        if isinstance(now[0].model, ZSVideoModel) or isinstance(now[0].model,
                                                                                                SAUDIOModel) or isinstance(
                                            now[0].model, SVDVideoModel):
                                            view_type = None
                                        else:
                                            if finalized[interaction]:
                                                view_type = "AgainAndUpscale"
                                        asyncio.run_coroutine_threadsafe(
                                            coro=edit_any_message(interaction, send_message, sendable_images, None,
                                                                  None),
                                            loop=client.loop)
                                        del sendable_images
                                    else:
                                        for prompt in now:
                                            if prompt.interaction == interaction:
                                                this_request = prompt
                                                break
                                        output_count = 0
                                        for image in images[interaction]:
                                            if isinstance(image, GenericOutput) and not isinstance(image,
                                                                                                   IntermediateOutput):
                                                output_count += 1
                                        current = 0
                                        for x in i.interactions:
                                            if x == interaction:
                                                current += 1
                                        progress = ((current * i.current) + (output_count * i.total[0])) * 100 / (
                                                i.total[0] * this_request.amount)
                                        send_message = str(round(progress, 2)) + "% " + str(
                                            round(time.time() - start_time, 2)) + "s"
                                        asyncio.run_coroutine_threadsafe(
                                            coro=edit_any_message(interaction, send_message, None, None, None),
                                            loop=client.loop)
            except Exception as e:
                model_passthrough = False
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(repr(e))
                with open("err.log", "a") as err_log:
                    err_log.write(
                        str(exc_type) + " " + str(fname) + " " + str(exc_tb.tb_lineno) + "\n" + str(repr(e)) + "\n")
                pass
        images = {}
        if run_queue != None and run_queue[0].model.path == now[0].model.path and model_passthrough:
            try:
                print(now[0].model.model.device.type)
            except: pass
            run_queue[0].model = now[0].model
        else:
            print("deleting model")
            now[0].model.del_model()
        del now
        gc.collect()
        torch.cuda.empty_cache()


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
            choices=list(str(x) for x in default_images.keys()),
            required=False,
            description="The model to use to generate the image",
        ),
        images: Optional[int] = discord.SlashOption(
            name="images",
            choices={"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10},
            required=False,
            description="How many images to generate (more will take longer)"
        ),
        images_multiplier: Optional[int] = discord.SlashOption(
            name="images_multiplier",
            choices={"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10},
            required=False,
            description="Sends this many messages with the same prompt"
        ),
):
    if interaction.user.id == 381983555930292224:
        global default_images
        global prompt_queue
        if not model: model = "sdxl-t"
        if not images: images = default_images[model]
        if not images_multiplier: images_multiplier = 1
        if not negative_prompt: negative_prompt = ""
        request = FactoryRequest(model=model_translations[model], prompt=prompt, negative_prompt=negative_prompt,
                                 amount=images,
                                 interaction=interaction)
        prompt_queue.append(request)
        await interaction.response.send_message("Generation has been queued.", view=AgainButton(request=request))
        # dont batch because model will be loaded to gpu anyways
        for idx in range(images_multiplier):
            if idx == 0: continue
            prompt_queue.append(
                FactoryRequest(model=model_translations[model], prompt=prompt, negative_prompt=negative_prompt,
                               amount=images,
                               interaction=(await interaction.channel.send("Generation has been queued.",
                                                                           view=AgainButton(request=request)))))
            # request can be reused since the button is request independent
    else:
        await interaction.response.send_message(
            "Dante is currently in a development state for Dante4. Please come back later")
        await interaction.channel.send(
            "<@381983555930292224> PUT ME DOWN NOW :rage: :rage: :face_with_symbols_over_mouth: :face_with_symbols_over_mouth: ")


live_sessions = {}
live_timestamp = {}


@client.slash_command(description="Enter a Dante Live session. Send this command to end your session.")
async def live(
        interaction: discord.Interaction,
        prompt: str,
):
    if interaction.user.id != 381983555930292224:
        await interaction.response.send("i wonder what this could be? dante4 soon!")
    else:
        await interaction.response.send_message("Live session ended.")
        try:
            del live_sessions[interaction.user]
        except:
            pass


@live.on_autocomplete("prompt")
async def live_prompt(interaction: discord.Interaction, prompt: str):
    if interaction.user.id != 381983555930292224:
        await interaction.response.send_autocomplete([])
    else:
        await interaction.response.send_autocomplete(["Prompt queued"])
        try:
            live_sessions[interaction.user]
        except:
            live_message = await interaction.channel.send("<@" + str(interaction.user.id) + ">Live session queued.")
            live_sessions[interaction.user] = live_message
            live_timestamp[interaction.user] = time.time()
        else:
            live_message = live_sessions[interaction.user]
            if live_timestamp[interaction.user] < (time.time() - 60):
                live_message = await interaction.channel.send("<@" + str(interaction.user.id) + ">\nLive session queued.")
                live_sessions[interaction.user] = live_message
        global prompt_queue
        live_timestamp[interaction.user] = time.time()
        if prompt and prompt != "Prompt queued":
            prompt_queue.append(FactoryRequest(model=model_translations["sdxl-t"], prompt=prompt,
                                               negative_prompt="",
                                               amount=5,
                                               interaction=live_message))


threading.Thread(target=model_factory, daemon=True).start()
threading.Thread(target=model_runner, daemon=True).start()
client.run(TOKEN)
