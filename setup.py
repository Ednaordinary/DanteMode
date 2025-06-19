import torch
from queue import Queue

import orchestrator

queue = orchestrator.start()

downloads = [
    ["black-forest-labs/FLUX.1-dev", "flux-dev", {"torch_dtype": torch.bfloat16}],
    ["black-forest-labs/FLUX.1-schnell", "flux-dev", {"text_encoder": None, "text_encoder_2": None, "torch_dtype": torch.bfloat16}]
]


for i in downloads:
    function = orchestrator.Function("download", i[0], i[1], **i[2])
    queue.put(function)

end = Queue()
queue.put(orchestrator.End(end))
end.get()
