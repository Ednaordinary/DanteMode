import random
import subprocess
import sys

import torchaudio
from diffusers import FluxTransformer2DModel
from optimum.quanto import quantize, freeze, qint4, qint8
from transformers import T5EncoderModel

from models.audio import SAUDIOModel
from models.generic import GenericModel, GenericOutput, RunStatus, Prompt, FinalOutput
from models.intermediate import IntermediateOutput, IntermediateOptimizedModel, IntermediateModel
from models.pasi import PASIModel
from models.sd import SDXLModel, SDXLTModel, SD3Model, SCASCModel, SDXLDSModel, SDXLJXModel, SDDSModel, SDXLDSLITModel
from models.flux import FLUXDevModel, FLUXDevTempModel
from diffusers.utils import numpy_to_pil, export_to_video
from dotenv import load_dotenv
from typing import Optional
import nextcord as discord
from PIL import Image
import threading
import asyncio
import torch
import vram
import time
import PIL
import gc
import io
import os

from models.upscale import LDMUpscaleModel
from models.video import ZSVideoModel, SVDVideoModel, SV3DVideoModel

torch.backends.cuda.matmul.allow_tf32 = True

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.AutoShardedClient(intents=intents)
prompt_queue = []
run_queue = None
current_model_path = None

# # The following is TEMPORARY until some form of quantized model can be quickly loaded from file.
# # Takes up a LOT of ram constantly

temp_flux_dev_transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer", revision="refs/pr/3", torch_dtype=torch.bfloat16)
temp_flux_dev_text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2",
                                                                torch_dtype=torch.bfloat16)
temp_flux_schnell_transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="transformer", revision="refs/pr/1", torch_dtype=torch.bfloat16)
temp_flux_schnell_text_encoder_2 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="text_encoder_2",
                                                                torch_dtype=torch.bfloat16)

def quantize_thread(object, name):
    print("Quantizing", name)
    try:
        quantize(object, qint8)
        #quantize(object, qint4, exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"])
        freeze(object)
    except Exception as e:
        print(repr(e))
    print("Done", name)
quant_threads = []
for quantable in [[temp_flux_dev_transformer, "dev transformer"], [temp_flux_dev_text_encoder_2, "dev text encoder"], [temp_flux_schnell_transformer, "schnell transformer"], [temp_flux_schnell_text_encoder_2, "schnell text encoder"]]:
    quant_threads.append(threading.Thread(target=quantize_thread, args=[quantable[0], quantable[1]]))
for thread in quant_threads:
    time.sleep(0.01) # just so text doesn't overlap
    thread.start()
for thread in quant_threads:
    thread.join()

model_translations = {
    "sd": IntermediateOptimizedModel(path="runwayml/stable-diffusion-v1-5", out_type="image", max_latent=50, steps=30,
                                     mini_vae="madebyollin/taesd"),
    "sd2": IntermediateOptimizedModel(path="stabilityai/stable-diffusion-2-1", out_type="image", max_latent=30,
                                      steps=30,
                                      mini_vae="madebyollin/taesd"),
    "sdxl": SDXLModel(path="stabilityai/stable-diffusion-xl-base-1.0", out_type="image", max_latent=10, steps=35,
                      mini_vae="madebyollin/taesdxl"),
    "sdxl-ds": SDXLDSModel(path="Lykon/dreamshaper-xl-1-0", out_type="image", max_latent=15, steps=35,
                           mini_vae="madebyollin/taesdxl"),
    "sdxl-ds-lit": SDXLDSLITModel(path="lykon/dreamshaper-xl-lightning", out_type="image", max_latent=10, steps=4,
                                  mini_vae="madebyollin/taesdxl"),
    "sdxl-jx": SDXLJXModel(path="RunDiffusion/Juggernaut-X-v10", out_type="image", max_latent=10, steps=35,
                           mini_vae="madebyollin/taesdxl"),
    "sdxl-t": SDXLTModel(path="stabilityai/sdxl-turbo", out_type="image", max_latent=100, steps=4),
    "sd-ds": SDDSModel(path="Lykon/dreamshaper-8", out_type="image", max_latent=50, steps=30,
                       mini_vae="madebyollin/taesd"),
    "sd3-m": SD3Model(path="stabilityai/stable-diffusion-3-medium-diffusers", out_type="image", max_latent=3, steps=35,
                      mini_vae="madebyollin/taesd3"),
    "scasc": SCASCModel(path="stabilityai/stable-cascade", out_type="image", max_latent=10, steps=20),
    "pa-si": PASIModel(path="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", out_type="image", max_latent=20, steps=35,
                       mini_vae="madebyollin/taesdxl"),
    "flux-d": FLUXDevTempModel(path="black-forest-labs/FLUX.1-dev", out_type="image", max_latent=3, steps=25, transformer=temp_flux_dev_transformer, text_encoder_2=temp_flux_dev_text_encoder_2, guidance_scale=3.5),
    "flux-s": FLUXDevTempModel(path="black-forest-labs/FLUX.1-schnell", out_type="image", max_latent=3, steps=2, transformer=temp_flux_schnell_transformer, text_encoder_2=temp_flux_schnell_text_encoder_2, guidance_scale=0.0),
    #"flux-d": FLUXDevModel(path="black-forest-labs/FLUX.1-dev", out_type="image", max_latent=1, steps=40, guidance_scale=3.5, local_path="flux-dev-fp8"),
    #"flux-s": FLUXDevModel(path="black-forest-labs/FLUX.1-schnell", out_type="image", max_latent=1, steps=4, guidance_scale=0.0, local_path="flux-schnell-fp8"),
    "s-video": SVDVideoModel(path="stabilityai/stable-video-diffusion-img2vid-xt-1-1", out_type="video-zs",
                             max_latent=1, steps=35, mini_vae="madebyollin/taesdxl"),
    "zs-video": ZSVideoModel(path="cerspense/zeroscope_v2_576w", out_type="video-zs", max_latent=1, steps=40),
    "s-audio": SAUDIOModel(path="stabilityai/stable-audio-open-1.0", out_type="s-audio", max_latent=5, steps=100),
    "s-3d": SV3DVideoModel(path="stabilityai/sv3d", out_type="video-zs",
                             max_latent=1, steps=35, mini_vae="madebyollin/taesdxl"),
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
    "scasc": 10,
    "pa-si": 10,
    "flux-d": 3,
    "flux-s": 10,
    "s-video": 1,
    "zs-video": 3,
    "s-audio": 3,
    #"s-3d": 1,
}
images = {}

#This may take a while. Be really patient.

def quickstart_models(model):
    # Doing this makes sure flux-s and flux-d will be ready to load
    this_model = model_translations[model]
    this_model.to("cpu")
    this_model.del_model()

quickstart_threads = []
for quickstart in [threading.Thread(target=quickstart_models, args=["flux-d"]), threading.Thread(target=quickstart_models, args=["flux-s"])]:
    quickstart_threads.append(quickstart)
    quickstart.start()
for quickstart in quickstart_threads:
    quickstart.join()

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
                if current_model_path == None:
                    print("loading model to gpu")
                    prompt_queue[0].model.to('cuda')
                    device = 'gpu'
                else:
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
                            coro=channel.send("Generation has been queued."),
                            loop=client.loop
                        ).result()
                        prompt_queue.append(FactoryRequest(model=model_translations["sdxl-t"], prompt=prompt,
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
            vram.allocate("Dante")
            async for i in vram.wait_for_allocation("Dante"):
                for request in now:
                    asyncio.run_coroutine_threadsafe(
                        coro=edit_any_message(request.interaction, "Waiting for " + i, None, None, None),
                        loop=client.loop)
            now[0].model.to('cuda')
            send_cuda_message = True
        else:
            if now[0].model.model.device.type != "cuda":
                vram.allocate("Dante")
                async for i in vram.wait_for_allocation("Dante"):
                    for request in now:
                        asyncio.run_coroutine_threadsafe(
                            coro=edit_any_message(request.interaction, "Waiting for " + i, None, None, None),
                            loop=client.loop)
                now[0].model.to("cuda")
                send_cuda_message = True
        start_time = time.time()
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
                    coro=edit_any_message(request.interaction, "Model loaded to gpu", None, None, None),
                    loop=client.loop)
            diffusing_amount += request.amount
        activity = discord.Activity(name="Diffusion", state="Diffusing " + str(diffusing_amount) + " images | " + str(
            len(prompt_queue)) + " requests in queue", type=discord.ActivityType.watching)
        asyncio.run_coroutine_threadsafe(
            coro=client.change_presence(activity=activity, status=discord.Status.online),
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
                                        tmp_image.thumbnail((256, 256))
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
                                    else:
                                        view_type = None
                                if isinstance(now[0].model, LDMUpscaleModel):
                                    asyncio.run_coroutine_threadsafe(coro=edit_any_message(interaction, str(len(
                                        sendable_images)) + " images upscaled in " + str(
                                        round(time.time() - start_time, 2)) + "s", [x for x in sendable_images],
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
                                                tmp_image.thumbnail((256, 256))
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
                                            else:
                                                view_type = None
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
            run_queue[0].model = now[0].model
        else:
            print("deleting model")
            now[0].model.del_model()
            vram.deallocate("Dante")
            asyncio.run_coroutine_threadsafe(
                coro=client.change_presence(activity=None, status=discord.Status.idle),
                loop=client.loop)
        model_path = now[0].model.path
        current_model_path = None
        del now
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Current memory: {torch.cuda.memory_allocated(device="cuda") / 1024 ** 3:.3f}GiB')
        # This log is purely for debugging purposes, all it stores is memory allocation and the last model at that time.
        with open("allocation.log", "a") as err_log:
            err_log.write(
                str(model_path) + f" | Post-run allocated memory: {torch.cuda.memory_allocated(device="cuda") / 1024 ** 3:.3f}GiB\n")
        del model_path


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
    global default_images
    global prompt_queue
    if not model: model = "flux-s"
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
        # request can be reused since the button is interaction independent


live_sessions = {}
live_timestamp = {}


@client.slash_command(description="Enter a Dante Live session. Send this command to end your session.")
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
    await interaction.response.send_autocomplete(["Prompt queued"])
    try:
        live_sessions[interaction.user]
    except:
        live_message = await interaction.channel.send("<@" + str(interaction.user.id) + ">\nLive session queued.")
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
threading.Thread(target=file_queuer, daemon=True).start()
client.run(TOKEN)
