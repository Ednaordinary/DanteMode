import time

startup_time = time.perf_counter()

import random
import subprocess
import sys

import torchaudio
#from diffusers import FluxTransformer2DModel
#from optimum.quanto import quantize, freeze, qint4, qint8
from transformers import T5EncoderModel

from models.audio import SAUDIOModel
from models.allegro import AllegroModel
from models.mochi import MochiModel
from models.hunyuan import HNModel
from models.generic import GenericModel, GenericOutput, RunStatus, Prompt, FinalOutput
from models.intermediate import IntermediateOutput, IntermediateOptimizedModel, IntermediateModel
from models.pasi import PASIModel
from models.sd import SDXLModel, SDXLTModel, SD3Model, SCASCModel, SDXLDSModel, SDXLJXModel, SDDSModel, SDXLDSLITModel
from models.flux import FLUXModel, unpack_flux_latents
from models.upscale import LDMUpscaleModel
from models.video import ZSVideoModel, SVDVideoModel, SV3DVideoModel, CogVideoModel #, PyramidFlowModel
from diffusers.utils import numpy_to_pil
from dotenv import load_dotenv
from typing import Optional
import nextcord as discord
from PIL import Image
#from numba import cuda as numba_cuda
import imageio as iio
import numpy as np
import threading
import asyncio
import psutil
import torch
import vram
import time
import PIL
import gc
import io
import os
import ctypes

libc = ctypes.CDLL("libc.so.6") # Needed for memory management

print("imports complete")

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

numba_device = numba_cuda.get_current_device()

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.AutoShardedClient(intents=intents, shard_count=5)
prompt_queue = []
edit_fix = {}
run_queue = None
current_model_path = None
startup_sent = False

model_translations = {
    "sd": (IntermediateOptimizedModel, dict(path="runwayml/stable-diffusion-v1-5", out_type="image", max_latent=50, steps=30,
                                     mini_vae="madebyollin/taesd")),
    "sd2": (IntermediateOptimizedModel, dict(path="stabilityai/stable-diffusion-2-1", out_type="image", max_latent=30,
                                      steps=30,
                                      mini_vae="madebyollin/taesd")),
    "sdxl": (SDXLModel, dict(path="stabilityai/stable-diffusion-xl-base-1.0", out_type="image", max_latent=10, steps=35,
                      mini_vae="madebyollin/taesdxl")),
    "sdxl-ds": (SDXLDSModel, dict(path="Lykon/dreamshaper-xl-1-0", out_type="image", max_latent=15, steps=35,
                           mini_vae="madebyollin/taesdxl")),
    "sdxl-ds-lit": (SDXLDSLITModel, dict(path="lykon/dreamshaper-xl-lightning", out_type="image", max_latent=10, steps=4,
                                  mini_vae="madebyollin/taesdxl")),
    "sdxl-jx": (SDXLJXModel, dict(path="RunDiffusion/Juggernaut-X-v10", out_type="image", max_latent=10, steps=35,
                           mini_vae="madebyollin/taesdxl")),
    "sdxl-t": (SDXLTModel, dict(path="stabilityai/sdxl-turbo", out_type="image", max_latent=100, steps=4)),
    "sd-ds": (SDDSModel, dict(path="Lykon/dreamshaper-8", out_type="image", max_latent=50, steps=30,
                       mini_vae="madebyollin/taesd")),
    "sd3-m": (SD3Model, dict(path="stabilityai/stable-diffusion-3.5-medium", out_type="image", max_latent=3, steps=28,
                      mini_vae="madebyollin/taesd3", guide=7.0)),
    "sd3-l": (SD3Model, dict(path="stabilityai/stable-diffusion-3.5-large", out_type="image", max_latent=3, steps=28, mini_vae="madebyollin/taesd3", guide=4.5)),
    "sd3-lt": (SD3Model, dict(path="stabilityai/stable-diffusion-3.5-large-turbo", out_type="image", max_latent=3, steps=4, mini_vae="madebyollin/taesd3", guide=0.0)),
    "scasc": (SCASCModel, dict(path="stabilityai/stable-cascade", out_type="image", max_latent=10, steps=20)),
    "pa-si": (PASIModel, dict(path="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", out_type="image", max_latent=20, steps=35,
                       mini_vae="madebyollin/taesdxl")),
    "flux-d": (FLUXModel, dict(path="black-forest-labs/FLUX.1-dev", out_type="image", max_latent=10, steps=30, guidance_scale=4.5, max_seq=512, transformerpath="flux-dev-transformer", res=1024, para=0.12, l_step=6)),
    "flux-s": (FLUXModel, dict(path="black-forest-labs/FLUX.1-schnell", out_type="image", max_latent=10, steps=4, guidance_scale=0.0, max_seq=256, transformerpath="flux-schnell-transformer", res=1024, para=0.6, l_step=2)),
    "moc-video": (MochiModel, dict(path="genmo/mochi-1-preview", out_type="video-zs", max_latent=1, steps=64, guidance=8.5, length=43, flavr_path="FLAVR.pth", dynamic_cfg=True)),
    "hn-video": (HNModel, dict(path="tencent/HunyuanVideo", transformerpath="hun-transformer", out_type="video-zs", max_latent=1, steps=30, guidance=6.0, length=45, flavr_path="FLAVR.pth", shift=7.0, para=0.06)),
    "fvhn-video": (HNModel, dict(path="tencent/HunyuanVideo", transformerpath="fvhn-transformer", out_type="video-zs", max_latent=1, steps=7, guidance=6.0, length=65, flavr_path="FLAVR.pth", shift=19.0, para=0.15)),
    "agl-video": (AllegroModel, dict(path="rhymes-ai/Allegro", out_type="video-zs", max_latent=1, steps=20, revision="refs/pr/2")),
    #"pf-video": PyramidFlowModel(path="models/pf/pyramid-flow", out_type="video-zs", max_latent=1, steps=28, cpu_offload=True, variant='diffusion_transformer_768p'),
    "cg-video": (CogVideoModel, dict(path="THUDM/CogVideoX-2b", out_type="video-zs", max_latent=1, steps=50, cpu_offload=False)),
    "cgl-video": (CogVideoModel, dict(path="THUDM/CogVideoX-5b", out_type="video-zs", max_latent=1, steps=50, cpu_offload=True)),
    "s-video": (SVDVideoModel, dict(path="stabilityai/stable-video-diffusion-img2vid-xt-1-1", out_type="video-zs",
                             max_latent=1, steps=35, mini_vae="madebyollin/taesdxl")),
    "zs-video": (ZSVideoModel, dict(path="cerspense/zeroscope_v2_576w", out_type="video-zs", max_latent=1, steps=60)),
    "s-audio": (SAUDIOModel, dict(path="stabilityai/stable-audio-open-1.0", out_type="s-audio", max_latent=5, steps=100)),
    "ldm-upscale": (LDMUpscaleModel, dict(path="CompVis/ldm-super-resolution-4x-openimages", out_type="image", max_latent=1,
                                  steps=40))
    #"s-3d": SV3DVideoModel(path="stabilityai/sv3d", out_type="video-zs",
    #                         max_latent=1, steps=35, mini_vae="madebyollin/taesdxl"),
}

default_images = {
    "sd": 10,
    "sd2": 10,
    "sdxl": 10,
    "sdxl-ds": 10,
    "sdxl-ds-lit": 10,
    "sdxl-jx": 10,
    "sdxl-t": 10,
    "sd-ds": 10,
    "sd3-m": 5,
    "sd3-l": 3,
    "sd3-lt": 3,
    "scasc": 10,
    "pa-si": 10,
    "flux-d": 3,
    "flux-s": 10,
    "hn-video": 1,
    "fvhn-video": 1,
    "moc-video": 1,
    #"fvmoc-video": 1,
    #"mocl-video": 1,
    "agl-video": 1,
    #"pf-video": 1,
    "cg-video": 1,
    "cgl-video": 1,
    "s-video": 1,
    "zs-video": 2,
    "s-audio": 3,
    #"s-3d": 1,
}
images = {}

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    libc.malloc_trim(0)

def error_handler(e, part):
    if e is not None:
        print("Error received:", e, part)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
        embed = discord.Embed()
        embed.add_field(name="Error in " + part, value=str(exc_type) + "\n" + str(fname) + "\n" + str(exc_tb.tb_lineno) + "\n" + repr(e), inline=True)
        embed.set_footer(text=f'Current vram: {torch.cuda.memory_allocated(device="cuda") / 1024 ** 3:.3f}GiB')
    else:
        embed = discord.Embed()
        embed.add_field(name="Dante",
                        value=part,
                        inline=True)
    try:
        channel = client.get_channel(int(os.getenv('DEBUG_CHANNEL')))
        asyncio.run_coroutine_threadsafe(
            coro=channel.send(embed=embed),
            loop=client.loop
        )
    except Exception as e:
        print(repr(e))


async def edit_any_message(message, content, files, view, request):
    global edit_fix
    if view == "AgainAndUpscale":
        try:
            request.interaction.context
        except:
            view = AgainAndUpscaleButton(request=request)
        else:
            if request and request.interaction.context != discord.InteractionContextType.guild:
                view = None
            else:
                view = AgainAndUpscaleButton(request=request)
    try:
        edit_fix[message]
    except:
        pass
    else:
        while edit_fix[message] == 0:
            time.sleep(0.01) # prevent message sending multiple times while waiting to send the first
        message = edit_fix[message]
        content = "**Something went wrong while editing the message! Here's a new one.**\n" + content
        content = content[:2000]
    for i in range(3):  # Sometimes we'll get mac address errors due to load balancing
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
    # ok so we know it definitely isn't a mac address error, let's actually do something about it
    print("Errored out a lot, trying to fix")
    edit_fix[message] = 0
    try:
        params = {"content": (("**Something went wrong while editing the message! Here's a new one.**\n" + content)[
                              :2000] if content != None else "**Something went wrong while editing the message! Here's a new one.**\n"),
                  "files": files, "view": view}
        params = {k: v for k, v in params.items() if v is not None}
        new_message = await message.channel.send(**params)
    except Exception as e:  # something's really wrong
        error_handler(e, "message editor")
        del edit_fix[message]
    else:
        edit_fix[message] = new_message

def export_to_video_bytes(fps, frames):
    request = iio.core.Request("<bytes>", mode="w", extension=".mp4")
    pyavobject = iio.plugins.pyav.PyAVPlugin(request)
    if isinstance(frames, np.ndarray):
        frames = (np.array(frames) * 255).astype('uint8')
    else:
        frames = np.array(frames)
    new_bytes = pyavobject.write(frames, codec="libx264", fps=fps)
    out_bytes = io.BytesIO(new_bytes)
    return out_bytes

async def ping_request_user(request):
    interaction = request.interaction
    user = interaction.user
    channel = interaction.channel
    await channel.send(f"<@{user.id}>, your request is complete")

class AgainButton(discord.ui.View):
    def __init__(self, *, timeout=None, request):
        super().__init__(timeout=timeout)
        self.amount = request.amount
        self.model = request.model_idx
        self.prompt = request.prompt
        self.negative_prompt = request.negative_prompt

    @discord.ui.button(label="Again", style=discord.ButtonStyle.primary)
    async def again_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        print(interaction.user)
        message = await interaction.channel.send("Queued.", view=self)
        global prompt_queue
        prompt_queue.append(
            FactoryRequest(model=model_translations[self.model][0](**model_translations[self.model][1]), model_idx=self.model, prompt=self.prompt,
                           negative_prompt=self.negative_prompt,
                           amount=self.amount,
                           interaction=message))
        button.style = discord.ButtonStyle.secondary
        await interaction.response.edit_message(view=self)


class AgainAndUpscaleButton(discord.ui.View):
    def __init__(self, *, timeout=None, request):
        super().__init__(timeout=timeout)
        self.amount = request.amount
        self.model = request.model_idx
        self.prompt = request.prompt
        self.negative_prompt = request.negative_prompt

    @discord.ui.button(label="Again", style=discord.ButtonStyle.primary)
    async def again_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        print(interaction.user)
        message = await interaction.channel.send("Queued.", view=self)
        global prompt_queue
        prompt_queue.append(
            FactoryRequest(model=model_translations[self.model][0](**model_translations[self.model][1]), model_idx=self.model, prompt=self.prompt,
                           negative_prompt=self.negative_prompt,
                           amount=self.amount,
                           interaction=message))
        button.style = discord.ButtonStyle.secondary
        await interaction.response.edit_message(view=self)

    @discord.ui.button(label="Upscale", style=discord.ButtonStyle.primary)
    async def upscale_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        print(interaction.user)
        message = await interaction.channel.send("Queued.")
        global prompt_queue
        images = []
        button.style = discord.ButtonStyle.secondary
        await interaction.response.edit_message(view=self)
        for attachment in interaction.message.attachments:
            image = await attachment.read()
            images.append(Image.open(io.BytesIO(image)).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS))
        prompt_queue.append(FactoryRequest(
            model=model_translations["ldm-upscale"][0](**model_translations["ldm-upscale"][1]), prompt=images, negative_prompt="",
            amount=len(interaction.message.attachments),
            interaction=message))


class FactoryRequest:
    def __init__(self, model, model_idx, prompt, negative_prompt, amount, interaction):
        self.model = model
        self.model_idx = model_idx
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.amount = amount
        self.interaction = interaction
        self.start_time = time.perf_counter()


def model_factory():
    global prompt_queue
    global run_queue
    global current_model_path
    global live_sessions
    while True:
        if prompt_queue != [] and run_queue != None:
            if prompt_queue[0].model_idx == run_queue[0].model_idx:
                run_queue.append(prompt_queue[0])
                prompt_queue.pop(0)
                flush()
        if prompt_queue != [] and run_queue == None:  # has to be reevaluated
            device = 'gpu'
            if not prompt_queue[0].model_idx == current_model_path:
                vram.allocate("Dante")
                if current_model_path == None and vram.isfirst("Dante"):
                    print("loading model to gpu")
                    try:
                        prompt_queue[0].model.to('cuda')
                    except Exception as e:
                        error_handler(e, "model factory (cuda)")
                        flush()
                    device = 'gpu'
                else:
                    print("loading model to cpu")
                    try:
                        prompt_queue[0].model.to('cpu')
                    except Exception as e:
                        error_handler(e, "model factory (cpu)")
                        flush()
                    device = 'cpu'
            tmp_queue = []
            tmp_path = prompt_queue[0].model_idx
            pop_amt = 0
            for prompt in prompt_queue:
                if not prompt.model_idx == tmp_path:
                    break
                tmp_queue.append(prompt)
                send_load_message = True
                for u, int in live_sessions.items():
                    if prompt.interaction == int:
                        send_load_message = False
                        break
                if send_load_message:
                    asyncio.run_coroutine_threadsafe(
                        coro=edit_any_message(prompt.interaction, "Loaded to " + device, None, None, None),
                        loop=client.loop
                    )
                pop_amt += 1
            for i in range(pop_amt): prompt_queue.pop(0)
            run_queue = tmp_queue
            if prompt:
                del prompt
            del tmp_queue, tmp_path
            flush()
        time.sleep(0.01)
    print("ended factory")


def file_queuer():
    global prompt_queue
    while True:
        overwrite = False
        with open("./queue.txt", "r") as file_queue:
            lines = file_queue.readlines()
            if [x for x in lines if x.strip() != ""] != []:
                overwrite = True
                for x in [x for x in lines if x.strip() != ""]:
                    prompt = x.split("|")
                    channel_id = int(prompt[0])
                    prompt = "".join(prompt[1:]).replace("\\n", "\n").strip()
                    channel = client.get_channel(channel_id)
                    if channel == None:
                        channel = asyncio.run_coroutine_threadsafe(
                            coro=client.fetch_channel(channel_id),
                            loop=client.loop
                        ).result()
                    if channel != None:
                        message = asyncio.run_coroutine_threadsafe(
                            coro=channel.send("Queued."),
                            loop=client.loop
                        ).result()
                        prompt_queue.append(FactoryRequest(model=model_translations["flux-s"][0](**model_translations["flux-s"][1]), model_idx="flux-s", prompt=prompt,
                                                           negative_prompt="",
                                                           amount=5,
                                                           interaction=message))
        if overwrite:
            with open("./queue.txt", 'w') as file_queue:
                pass
        time.sleep(0.01)


async def async_model_runner():
    global prompt_queue
    global run_queue
    global images
    global current_model_path
    while True:
        while not run_queue:
            time.sleep(0.01)
        model_passthrough = True
        now = run_queue
        run_queue = None
        current_model_path = now[0].model_idx
        send_cuda_message = False
        finalized = {}
        updated = {}
        vram.allocate("Dante") # Excess vram allocations can and should happen.
        flush()
        try:
            now[0].model.model.device
        except:
            async for i in vram.wait_for_allocation("Dante"):
                for request in now:
                    asyncio.run_coroutine_threadsafe(
                        coro=edit_any_message(request.interaction, "Waiting for " + i, None, None, None),
                        loop=client.loop)
                del request
            now[0].model.to('cuda')
            send_cuda_message = True
        else:
            if now[0].model.model.device.type != "cuda":
                async for i in vram.wait_for_allocation("Dante"):
                    for request in now:
                        asyncio.run_coroutine_threadsafe(
                            coro=edit_any_message(request.interaction, "Waiting for " + i, None, None, None),
                            loop=client.loop)
                now[0].model.to("cuda")
                send_cuda_message = True
        start_time = time.perf_counter()
        prompts = []
        diffusing_amount = 0
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
                    coro=edit_any_message(request.interaction, "Loaded to gpu", None, None, None),
                    loop=client.loop)
            diffusing_amount += request.amount
        del request
        activity = discord.Activity(name="Diffusion", state="Diffusing " + str(diffusing_amount) + " request", type=discord.ActivityType.watching)
        asyncio.run_coroutine_threadsafe(
            coro=client.change_presence(activity=activity, status=discord.Status.online),
            loop=client.loop)
        limiter = time.perf_counter()
        with torch.no_grad(): #torch.inference_mode():
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
                                del prompt
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
                                            #video_path = str(random.randint(1, 10000000)) + ".mp4"
                                            #fps = 24 if isinstance(now[0].model, PyramidFlowModel) else (14 if (isinstance(now[0].model, AllegroModel) or isinstance(now[0].model, MochiModel)) else 7)
                                            fps = 14 if isinstance(now[0].model, AllegroModel) else (60 if isinstance(now[0].model, MochiModel) or isinstance(now[0].model, HNModel) else 7)
                                            videobn = export_to_video_bytes(fps, image.output)
                                            videobn.seek(0)
                                            sendable_images[image.prompt.index] = discord.File(fp=videobn, filename=str(
                                                image.prompt.index) + ".mp4")
                                            #os.remove(video_path)
                                            #os.remove("redo-" + video_path)
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
                                        flush()
                                        tmp_image = numpy_to_pil((tmp_image / 2 + 0.5).permute(1, 2, 0).float().numpy())[0]
                                        imagebn = io.BytesIO()
                                        tmp_image.thumbnail((256, 256))
                                        tmp_image.save(imagebn, format='JPEG', quality=80)
                                        imagebn.seek(0)
                                        if images[interaction][image.prompt.index] == image:
                                            images[interaction][image.prompt.index].output = tmp_image
                                        flush()
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
                                    #if prompt.negative_prompt != "":
                                    #    send_message = str(len(sendable_images)) + " images of '" + str(
                                    #        prompt.prompt) + "' (negative: '" + str(
                                    #        prompt.negative_prompt) + "') in " + str(
                                    #        round(time.perf_counter() - start_time, 2)) + "s"
                                    #    print(prompt.negative_prompt)
                                    #    print(send_message)
                                    #    send_message = send_message[:2000]
                                    #else:
                                    #    send_message = str(len(sendable_images)) + " images of '" + str(
                                    #        prompt.prompt) + "' in " + str(round(time.perf_counter() - start_time, 2)) + "s"
                                    #    send_message = send_message[:2000]
                                    time_format = time.perf_counter() - start_time
                                    send_message = (str(int(time_format // 60)) + "m " + str(int(time_format % 60)) + "s") if int(time_format) > 60 else ("1m" if int(time_format) == 60 else str(int(time_format)) + "s")
                                else:
                                    send_message = None
                                if now[0].model.out_type == "video-zs" or now[0].model.out_type == "s-audio":
                                    view_type = None
                                else:
                                    if finalized[interaction] and hasattr(interaction, "context") and interaction.context == discord.InteractionContextType.guild:
                                        view_type = "AgainAndUpscale"
                                    else:
                                        view_type = None
                                if isinstance(now[0].model, LDMUpscaleModel):
                                    asyncio.run_coroutine_threadsafe(coro=edit_any_message(interaction, str(len(
                                        sendable_images)) + " images upscaled in " + str(
                                        round(time.perf_counter() - start_time, 2)) + "s", [x for x in sendable_images],
                                                                                           None, None),
                                                                     loop=client.loop)
                                else:
                                    asyncio.run_coroutine_threadsafe(
                                        coro=edit_any_message(interaction, send_message, sendable_images, view_type,
                                                              this_request), loop=client.loop)
                            del this_request
                    if isinstance(i, IntermediateOutput):
                        images[i.prompt.interaction][i.prompt.index] = i
                        updated[i.prompt.interaction] = True
                    if isinstance(i, RunStatus):
                        if limiter + 1.0 < time.perf_counter():
                            limiter = time.perf_counter()
                            for interaction in list(set(i.interactions)):
                                if not finalized[interaction]:
                                    if updated[interaction]:
                                        updated[interaction] = False
                                        for prompt in now:
                                            if prompt.interaction == interaction:
                                                this_request = prompt
                                                break
                                        del prompt
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
                                                    #video_path = str(random.randint(1, 10000000)) + ".mp4"
                                                    #fps = 24 if isinstance(now[0].model, PyramidFlowModel) else (14 if (isinstance(now[0].model, AllegroModel) or isinstance(now[0].model, MochiModel)) else 7)
                                                    fps = 14 if isinstance(now[0].model, AllegroModel) else (60 if isinstance(now[0].model, MochiModel) or isinstance(now[0].model, HNModel) else 7)
                                                    videobn = export_to_video_bytes(fps, image.output)
                                                    videobn.seek(0)
                                                    sendable_images[image.prompt.index] = discord.File(fp=videobn,
                                                                                                       filename=str(
                                                                                                           image.prompt.index) + ".mp4")
                                                    #os.remove(video_path)
                                                    #os.remove("redo-" + video_path)
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
                                                if isinstance(now[0].model, FLUXModel):
                                                    print(image.output.shape)
                                                    tmp_image = unpack_flux_latents(image.output.unsqueeze(0), now[0].model.res, now[0].model.res, now[0].model.model.vae_scale_factor)
                                                    tmp_image = ((tmp_image / now[0].model.model.vae.config.scaling_factor) + now[0].model.model.vae.config.shift_factor)
                                                    print(tmp_image.shape)
                                                    tmp_image = tmp_image[:,:,::3,::3]
                                                    print(tmp_image.shape)
                                                else:
                                                    tmp_image = image.output.unsqueeze(0)
                                                    tmp_image = tmp_image[:,:,::3,::3]
                                                print(tmp_image.shape)
                                                tmp_image = now[0].model.mini_vae.decode(tmp_image).sample[0]
                                                print(tmp_image.shape)
                                                tmp_image = tmp_image.to('cpu', non_blocking=False)
                                                flush()
                                                tmp_image = numpy_to_pil((tmp_image / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).float().numpy())[0]
                                                imagebn = io.BytesIO()
                                                tmp_image.thumbnail((512, 512))
                                                tmp_image.save(imagebn, format='JPEG', quality=80)
                                                imagebn.seek(0)
                                                if images[interaction][image.prompt.index] == image:
                                                    images[interaction][image.prompt.index].output = tmp_image
                                                flush()
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
                                        eta_calc = int(((100 / progress) * (time.perf_counter() - start_time)) - (time.perf_counter() - start_time))
                                        eta_calc = (str(int(eta_calc // 60)) + "m " + str(int(eta_calc % 60)) + "s") if int(eta_calc) > 60 else ("1m" if int(eta_calc) == 60 else str(int(eta_calc)) + "s")
                                        time_format = time.perf_counter() - start_time
                                        time_format = (str(int(time_format // 60)) + "m " + str(int(time_format % 60)) + "s") if int(time_format) > 60 else ("1m" if int(time_format) == 60 else str(int(time_format)) + "s")
                                        send_message = str(round(progress, 2)) + "% " + time_format + "\neta " + eta_calc
                                        if now[0].model.out_type == "video-zs" or now[0].model.out_type == "s-audio":
                                            view_type = None
                                        else:
                                            if finalized[interaction] and hasattr(interaction, "context") and interaction.context == discord.InteractionContextType.guild:
                                                view_type = "AgainAndUpscale"
                                            else:
                                                view_type = None
                                        asyncio.run_coroutine_threadsafe(
                                            coro=edit_any_message(interaction, send_message, sendable_images, None,
                                                                  None),
                                            loop=client.loop)
                                    else:
                                        for prompt in now:
                                            if prompt.interaction == interaction:
                                                this_request = prompt
                                                break
                                        del prompt
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
                                        eta_calc = int(((100 / progress) * (time.perf_counter() - start_time)) - (time.perf_counter() - start_time))
                                        eta_calc = (str(int(eta_calc // 60)) + "m " + str(int(eta_calc % 60)) + "s") if int(eta_calc) > 60 else ("1m" if int(eta_calc) == 60 else str(int(eta_calc)) + "s")
                                        time_format = time.perf_counter() - start_time
                                        time_format = (str(int(time_format // 60)) + "m " + str(int(time_format % 60)) + "s") if int(time_format) > 60 else ("1m" if int(time_format) == 60 else str(int(time_format)) + "s")
                                        send_message = str(round(progress, 2)) + "% " + time_format + "\neta " + eta_calc
                                        asyncio.run_coroutine_threadsafe(
                                            coro=edit_any_message(interaction, send_message, None, None, None),
                                            loop=client.loop)
                                        del this_request
            except Exception as e:
                for request in now:
                    asyncio.run_coroutine_threadsafe(
                        coro=edit_any_message(request.interaction, "(Something went wrong)", None, None, None), loop=client.loop)
                del request
                flush()
                model_passthrough = False
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                error_handler(e, "model runner")
                with open("err.log", "a") as err_log:
                    err_log.write(
                        str(exc_type) + " " + str(fname) + " " + str(exc_tb.tb_lineno) + "\n" + str(repr(e)) + "\n")
                pass
                #raise
        for request in now:
            if (time.perf_counter() - request.start_time) > (10 * 60):
                asyncio.run_coroutine_threadsafe(
                    coro=ping_request_user(request),
                    loop=client.loop)
        del request
        model_path = now[0].model_idx
        real_model_path = now[0].model.path
        if run_queue != None and run_queue[0].model_idx == now[0].model_idx and model_passthrough:
            run_queue[0].model = now[0].model
        else:
            print("deleting model")
            now[0].model.del_model()
            del now[0].model
            vram.deallocate("Dante")
            asyncio.run_coroutine_threadsafe(
                coro=client.change_presence(activity=None, status=discord.Status.idle),
                loop=client.loop)
        current_model_path, i, limiter, prompt, this_request, request = None, None, None, None, None, None
        del now, i, limiter, prompts, this_request, request
        finalized = {}
        updated = {}
        images = {}
        flush()
        print("Garbage:", gc.garbage)
        #numba_device.reset()
        print(f'Current vram: {torch.cuda.memory_allocated(device="cuda") / 1024 ** 3:.3f}GiB')
        print(f'Current memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.3f}GiB')
        # This log is purely for debugging purposes, all it stores is memory allocation and the last model at that time.
        with open("allocation.log", "a") as err_log:
            err_log.write(
                str(real_model_path) + f" | Post-run allocated memory: {torch.cuda.memory_allocated(device="cuda") / 1024 ** 3:.3f}GiB\n")
        del model_path
    print("exiting model runner")
    threading.Thread(target=mem_test).start()

def model_runner():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_model_runner())

def latency_checker():
    latency_sent = False
    global startup_sent
    while not startup_sent:
        time.sleep(0.01)
    time.sleep(20) # latency will be really high for the first little bit
    while True:
        time.sleep(10)
        if client.latency != []:
            for shard, latency in client.latencies:
                if latency > 0.5:
                    error_handler(None, "Warning!\nHigh latency: " + str(round(latency, 5)) + "s (shard id " + str(shard) + ")")
                    latency_sent = True
            if latency_sent:
                time.sleep(60)
                latency_sent = False

@client.event
async def on_ready():
    #print("Visible for: " + ", ".join(x.name for x in client.users))
    print(len(client.users), "total users")
    print(f'{client.user.name} has connected to Discord!')
    global startup_sent
    if not startup_sent:
        error_handler(None, "Startup complete\nStartup time: " + str(round(time.perf_counter() - startup_time, 2)) + "s")
        startup_sent = True

@client.event
async def on_error(event, args, kwargs):
    print(repr(event))
    print(args)
    print(kwargs)
    error_handler(event, "client loop")


@client.slash_command(description="Generates an image from the prompt", integration_types=[discord.IntegrationType.user_install, discord.IntegrationType.guild_install], contexts=[discord.InteractionContextType.guild, discord.InteractionContextType.bot_dm, discord.InteractionContextType.private_channel])
async def generate(
        interaction: discord.Interaction,
        prompt: str = discord.SlashOption(
            name="prompt",
            required=True,
            description="The prompt to generate off of",
        ),
        negative_prompt: Optional[str] = discord.SlashOption(
            name="negative_prompt",
            required=False,
            description="The negative prompt to generate off of",
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
):
    print(interaction.user)
    global default_images
    global prompt_queue
    if not model: model = "flux-s"
    if not images: images = default_images[model]
    if not negative_prompt: negative_prompt = ""
    request = FactoryRequest(model=model_translations[model][0](**model_translations[model][1]), model_idx=model, prompt=prompt, negative_prompt=negative_prompt,
                             amount=images,
                             interaction=interaction)
    if interaction.context == discord.InteractionContextType.guild:
        message = await interaction.response.send_message("Queued.", view=AgainButton(request=request))
    else:
        message = await interaction.response.send_message("Queued.")
    request.message = message
    prompt_queue.append(request)
    # dont batch because model will be loaded to gpu anyways


live_sessions = {}
live_timestamp = {}


@client.slash_command(description="Type to enter a Dante Live session. Send the command to end your session.", integration_types=[discord.IntegrationType.guild_install], contexts=[discord.InteractionContextType.guild, discord.InteractionContextType.bot_dm])
async def live(
        interaction: discord.Interaction,
        prompt: str,
):
    await interaction.response.send_message("Live session ended.")
    try:
        del live_sessions[interaction.user]
    except:
        pass


@live.on_autocomplete("prompt")
async def live_prompt(interaction: discord.Interaction, prompt: str):
    print(interaction.user)
    await interaction.response.send_autocomplete(["Prompt queued"])
    try:
        live_sessions[interaction.user]
    except:
        live_message = await interaction.channel.send("<@" + str(interaction.user.id) + ">\nLive session queued.")
        live_sessions[interaction.user] = live_message
        live_timestamp[interaction.user] = time.perf_counter()
    else:
        live_message = live_sessions[interaction.user]
        if live_timestamp[interaction.user] < (time.perf_counter() - 60):
            live_message = await interaction.channel.send("<@" + str(interaction.user.id) + ">\nLive session queued.")
            live_sessions[interaction.user] = live_message
    global prompt_queue
    live_timestamp[interaction.user] = time.perf_counter()
    if prompt and prompt != "Prompt queued":
        prompt_queue.append(FactoryRequest(model=model_translations["sdxl-t"][0](**model_translations["sdxl-t"][1]), model_idx="sdxl-t", prompt=prompt,
                                           negative_prompt="",
                                           amount=5,
                                           interaction=live_message))


threading.Thread(target=model_factory, daemon=True).start()
threading.Thread(target=model_runner, daemon=True).start()
threading.Thread(target=file_queuer, daemon=True).start()
#threading.Thread(target=latency_checker, daemon=True).start()
#numba_device.reset()
print("Definitions complete, starting connect")
client.run(TOKEN)
