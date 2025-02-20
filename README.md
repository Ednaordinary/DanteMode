# Dante4

## Ever wanted to efficiently run 16 diffusion models through a discord bot?


https://github.com/user-attachments/assets/020f5db2-f259-4c2a-8e4f-c7a40fd0e7b6



## Nitty gritty / Fully Featured

Dante is a framework for concurrent execution and resource management between image/video/audio generation and discord. Dante creates a queue for requests and executes them on a separate thread, then updates the original request message with the generated image/s. Dante uses many tricks to provide the user with the best experience:

- Model Passthrough: When a request with the same model currently being used is enqueued, Dante will keep that model on the gpu and reuse it, removing reload latency.

- Model Preloading: When a request with a different model than what's currently being loaded is enqueued, Dante will load that model to the cpu, so it can be quickly loaded to the gpu once the current request is done, instead of reading from disk (slow).

- Latent Previewing: With supported models, Dante sends previews of the images as they are being generated. This is done asynchronously from the main generation process, so it goes just as fast.

- Prompt Batching: When multiple requests with the same model are enqueued in a row, Dante will generate as many of these requests as it can at once, cutting down on overall compute time.

- Asynchronous Framework: Dante's discord handler, model factory, and model runner are entirely separate, allowing each to perform asynchronously of one another to reduce blocking to the highest possible level.

- Embedded Processes: All models run directly in the main script, meaning there is no import time for torch or other large packages.

- Live Status: For all models, both with and without latent previewing, Dante provides the current percent and time it is on in the generation process.

- Exception Handling: If something goes wrong during the generation process, Dante doesn't break as a whole.

- DanteLive: This is an experience built on top of Dante4 which allows the user to receive images as they are typing the prompt. By using all other optimizations, DanteLive reduces latency to mere seconds, and in some cases, less than a second.

## Spec details

The version of Dante4 provided in this repo is tuned to run on a 24GB vram gpu with 64GB ram (32GB ram should work fine though).

## Run yourself

It's built to run like any other generic discord.py bot:

- Make a virtual environment with a tool like python-venv: `python3 -m venv venv && cd venv && source bin/activate`
- Install the packages in requirements.txt: `pip install -r requirements.txt`
- Put your DISCORD_TOKEN in .env
- Run: `python3 ./main.py`

If you have [Maw](https://github.com/Ednaordinary/MawDiscord) cloned and installed in the same directory as the Dante repo, they can work together dynamically to automatically manage vram and even allow Maw to prompt Dante.

## Use yourself

Once in a discord server with a decent amount of permissions, just use /generate or /live!
