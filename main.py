import sys

from models.generic import GenericModel, GenericOutput, RunStatus, Prompt, FinalOutput
from models.intermediate import IntermediateOutput, IntermediateOptimizedModel, IntermediateModel
from models.sd import SDXLModel, SDXLTModel, SD3Model
from models.optimized import OptimizedModel
from diffusers.utils import numpy_to_pil
from dotenv import load_dotenv
from typing import Optional
import nextcord as discord
import threading
import asyncio
import torch
import time
import PIL
import gc
import io
import os

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
    "sd2": IntermediateOptimizedModel(path="stabilityai/stable-diffusion-2-1", out_type="image", max_latent=30, steps=30,
                                     mini_vae="madebyollin/taesd"),
    "sdxl": SDXLModel(path="stabilityai/stable-diffusion-xl-base-1.0", out_type="image", max_latent=15, steps=35,
                      mini_vae="madebyollin/taesdxl"),
    "sdxl-t": SDXLTModel(path="stabilityai/sdxl-turbo", out_type="image", max_latent=100, steps=4),
    "sd3-m": SD3Model(path="stabilityai/stable-diffusion-3-medium-diffusers", out_type="image", max_latent=10, steps=35,
                      mini_vae="madebyollin/taesd3"),
}
default_images = {
    "sd": 10,
    "sd2": 10,
    "sdxl": 10,
    "sdxl-t": 10,
    "sd3-m": 5,
}
images = {}


async def edit_any_message(message, content, files):
    try:
        if isinstance(message, discord.Interaction):
            if content == None:
                await message.edit_original_message(files=files)
            elif files == None:
                await message.edit_original_message(content=content)
            else:
                await message.edit_original_message(content=content, files=files)
        else:
            if content == None:
                await message.edit(files=files)
            elif files == None:
                await message.edit(content)
            else:
                await message.edit(content, files=files)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
        pass


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


def model_factory():
    global prompt_queue
    global run_queue
    global current_model_path
    while True:
        if prompt_queue != [] and run_queue != None:
            if prompt_queue[0].model.path == run_queue[0].model.path:
                run_queue.append(prompt_queue[0])
                prompt_queue.pop(0)
        if prompt_queue != []:
            print(prompt_queue)
        if prompt_queue != [] and run_queue == None:  # has to be reevaluated
            device = 'gpu'
            if not prompt_queue[0].model.path == current_model_path:
                prompt_queue[0].model.to('cpu')
                device = 'cpu'
            tmp_queue = []
            tmp_path = prompt_queue[0].model.path
            pop_amt = 0
            for prompt in prompt_queue:
                if not prompt.model.path == tmp_path:
                    break
                tmp_queue.append(prompt)
                #asyncio.run_coroutine_threadsafe(
                #    coro=prompt.interaction.edit_original_message(content="Model loaded to " + device), loop=client.loop)
                asyncio.run_coroutine_threadsafe(
                    coro=edit_any_message(prompt.interaction, "Model loaded to " + device, None), loop=client.loop
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
    while True:
        while not run_queue:
            time.sleep(0.01)
        now = run_queue
        run_queue = None
        current_model_path = now[0].model.path
        now[0].model.to('cuda')
        start_time = time.time()
        prompts = []
        for request in now:
            for i in range(request.amount):
                prompts.append(Prompt(prompt=request.prompt, negative_prompt=request.negative_prompt,
                                      interaction=request.interaction, index=i, parent_amount=request.amount))
            images[request.interaction] = [None] * request.amount
            finalized[request.interaction] = False
            #asyncio.run_coroutine_threadsafe(coro=request.interaction.edit_original_message(content="Model loaded to gpu"), loop=client.loop)
            asyncio.run_coroutine_threadsafe(coro=edit_any_message(request.interaction, "Model loaded to gpu", None),
                                             loop=client.loop)
        limiter = time.time()
        with torch.no_grad():
            async for i in now[0].model.call(prompts):
                if isinstance(i, FinalOutput):
                    for output in i.outputs:
                        images[output.prompt.interaction][output.prompt.index] = output
                    for interaction in list(set([x.prompt.interaction for x in i.outputs])):
                        print(interaction)
                        #if not finalized[interaction]:
                        if True:
                            print("unfinalized")
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
                                        sendable_images[image.prompt.index] = imagebn
                                    else:
                                        for_decoding.append(image)
                                        print(image.output.shape)
                            if for_decoding != None:
                                for image in for_decoding:
                                    tmp_image = now[0].model.mini_vae.decode(image.output).sample
                                    tmp_image = tmp_image.to('cpu', non_blocking=False)
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    tmp_image = numpy_to_pil((tmp_image / 2 + 0.5).permute(1, 2, 0).numpy())[0]
                                    imagebn = io.BytesIO()
                                    # tmp_image.show(
                                    #    title=image.prompt.prompt + str(image.prompt.index))  # for debugging indexing
                                    tmp_image.resize((128, 128)).save(imagebn, format='JPEG', quality=80)
                                    imagebn.seek(0)
                                    if images[interaction][image.prompt.index] == image:
                                        images[interaction][image.prompt.index].output = tmp_image
                                    del image.output
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    sendable_images[image.prompt.index] = imagebn
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
                                        prompt.prompt) + "' (negative: '" + str(prompt.negative_prompt) + "') in " + str(
                                        round(time.time() - start_time, 2)) + "s"
                                else:
                                    send_message = str(len(sendable_images)) + " images of '" + str(
                                        prompt.prompt) + "' in " + str(round(time.time() - start_time, 2)) + "s"
                            else:
                                send_message = None
                            #asyncio.run_coroutine_threadsafe(
                            #    coro=interaction.edit_original_message(content=send_message,
                            #                                                    files=[discord.File(fp=x, filename=str(idx) + ".jpg") for idx, x in enumerate(sendable_images)]), loop=client.loop)
                            asyncio.run_coroutine_threadsafe(coro=edit_any_message(interaction, send_message, [
                                discord.File(fp=x, filename=str(idx) + ".jpg") for idx, x in
                                enumerate(sendable_images)]), loop=client.loop)
                            del sendable_images
                if isinstance(i, IntermediateOutput):
                    images[i.prompt.interaction][i.prompt.index] = i
                if isinstance(i, RunStatus):
                    if limiter + 2.0 < time.time():
                        limiter = time.time()
                        for interaction in list(set(i.interactions)):
                            print(interaction)
                            if not finalized[interaction]:
                                print("unfinalized")
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
                                            sendable_images[image.prompt.index] = imagebn
                                        else:
                                            for_decoding.append(image)
                                if for_decoding != None:
                                    for image in for_decoding:
                                        print(image.prompt.index)
                                        tmp_image = now[0].model.mini_vae.decode(image.output).sample
                                        tmp_image = tmp_image.to('cpu', non_blocking=False)
                                        gc.collect()
                                        torch.cuda.empty_cache()
                                        tmp_image = numpy_to_pil((tmp_image / 2 + 0.5).permute(1, 2, 0).numpy())[0]
                                        imagebn = io.BytesIO()
                                        #tmp_image.show(
                                        #    title=image.prompt.prompt + str(image.prompt.index))  # for debugging indexing
                                        tmp_image.resize((128, 128)).save(imagebn, format='JPEG', quality=80)
                                        imagebn.seek(0)
                                        if images[interaction][image.prompt.index] == image:
                                            images[interaction][image.prompt.index].output = tmp_image
                                        del image.output
                                        gc.collect()
                                        torch.cuda.empty_cache()
                                        sendable_images[image.prompt.index] = imagebn
                                sendable_images = [x for x in sendable_images if x != None]
                                output_count = 0
                                for image in images[interaction]:
                                    if isinstance(image, GenericOutput) and not isinstance(image, IntermediateOutput):
                                        output_count += 1
                                if output_count == len(images[interaction]):
                                    finalized[interaction] = True
                                    print("setting finalized")
                                current = 0
                                for x in i.interactions:
                                    if x == interaction:
                                        current += 1
                                del x
                                progress = ((current * i.current) + (output_count * i.total[0])) * 100 / (
                                        i.total[0] * this_request.amount)
                                send_message = str(round(progress, 2)) + "% " + str(
                                    round(time.time() - start_time, 2)) + "s"
                                #asyncio.run_coroutine_threadsafe(
                                #    coro=interaction.edit_original_message(content=send_message,
                                #                                           files=[discord.File(fp=x, filename=str(idx) + ".jpg")
                                #                                                  for idx, x in enumerate(sendable_images)]),
                                #    loop=client.loop)
                                asyncio.run_coroutine_threadsafe(coro=edit_any_message(interaction, send_message, [
                                    discord.File(fp=x, filename=str(idx) + ".jpg")
                                    for idx, x in enumerate(sendable_images)]), loop=client.loop)
                                del sendable_images
        images = {}
        if run_queue != None and run_queue[0].model.path == now[0].model.path:
            print(run_queue)
            run_queue[0].model = now[0].model
        else:
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
        await interaction.response.send_message("Generation has been queued.")
        messages = [interaction]
        for idx in range(images_multiplier):  # waiting for this to complete means we can batch prompts
            if idx == 0: continue
            messages.append(await interaction.channel.send("Generation has been queued."))
        for message in messages:
            prompt_queue.append(
                FactoryRequest(model=model_translations[model], prompt=prompt, negative_prompt=negative_prompt,
                               amount=images,
                               interaction=message))
    else:
        await interaction.response.send_message(
            "Dante is currently in a development state for Dante4. Please come back later")
        await interaction.channel.send(
            "<@381983555930292224> PUT ME DOWN NOW :rage: :rage: :face_with_symbols_over_mouth: :face_with_symbols_over_mouth: ")


threading.Thread(target=model_factory, daemon=True).start()
threading.Thread(target=model_runner, daemon=True).start()
client.run(TOKEN)
