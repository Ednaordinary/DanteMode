import torch
from queue import Queue

import orchestrator

queue = orchestrator.start()

downloads = [
    ["black-forest-labs/FLUX.1-dev", "flux-dev", ["ae.safetensors", "flux1-dev.safetensors"], {"torch_dtype": torch.bfloat16}],
    ["black-forest-labs/FLUX.1-schnell", "flux-schnell", ["ae.safetensors", "flux1-schnell.safetensors"], {"text_encoder": None, "text_encoder_2": None, "torch_dtype": torch.bfloat16}]
]


for i in downloads:
    function = orchestrator.Function("download", i[0], i[1], i[2], **i[3])
    queue.put(function)

end = Queue()
queue.put(orchestrator.End(end))
end.get()
