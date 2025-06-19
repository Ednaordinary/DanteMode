import modal


class Orchestrator:
    image = modal.Image.from_registry("nvidia/cuda:12.9.0-runtime-ubuntu24.04")
    app = modal.App("Dante")

    def __init__(self):
        pass

    @app.function()
    def meow(self):
        print("meow from silly location")

    @app.function(gpu="T4", image=image)
    def get_embeds(self, pipeline):
        pass

    def main(self):
        self.meow.remote()

    def run(self):
        with modal.enable_output():
            with self.app.run():
                self.main()
