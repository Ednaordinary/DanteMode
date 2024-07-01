import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
class SAUDIOModel(GenericModel):
    def to(self, device):
        try:
            self.model
        except:
            self.model, config = get_pretrained_model(self.path)
        else:
            if not self.model:
                self.model, config = get_pretrained_model(self.path)
        self.model = self.model.to(device)
        self.model.vae.enable_slicing()
        self.sample_rate = config["sample_rate"]
        self.sample_size = config["sample_size"]

    async def call(self, prompts):
        self.to("cuda")

        def threaded_model(model, prompts, negative_prompts, steps, callback):
            try:
                #self.out = model(prompts, negative_prompt=negative_prompts, num_inference_steps=steps,
                #                 callback=callback, callback_steps=1)
                self.out = generate_diffusion_cond()
            except:
                self.out = [[]]
                pass

        def progress_callback(i, t, latents):
            self.step = i

        for i in range(0, len(prompts), self.max_latent):
            model_thread = threading.Thread(target=threaded_model,
                                            args=[self.model, [x.prompt for x in prompts[i:i + self.max_latent]],
                                                  [x.negative_prompt for x in prompts[i:i + self.max_latent]],
                                                  self.steps, progress_callback])
            model_thread.start()
            step = 0
            self.step = 0
            while model_thread.is_alive():
                if step != self.step:
                    yield RunStatus(current=self.step,
                                    total=self.steps,
                                    interactions=[x.interaction for x in prompts[i:i + self.max_latent]])
                    step = self.step
                time.sleep(0.01)
            outputs = []
            for idx, out in enumerate(self.out[0]):
                outputs.append(
                    GenericOutput(output=out, out_type=self.out_type, prompt=prompts[i:i + self.max_latent][idx]))
            yield FinalOutput(outputs=outputs)