import asyncio
import torch

import orchestrator

downloads = [
    ["black-forest-labs/FLUX.1-dev", {"torch_dtype": torch.bfloat16}],
    ["black-forest-labs/FLUX.1-schnell", {"text_encoder": None, "text_encoder_2": None, "torch_dtype": torch.bfloat16}]
]


async def setup():
    for i in downloads:
        await asyncio.gather(*[orchestrator.download(i[0], i[1]) for i in downloads])


loop = asyncio.new_event_loop()
loop.run_until_complete(setup())
