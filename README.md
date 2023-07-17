# DanteMode

Dante is a framework for concurrent execution and management between image generation and discord. Dante creates a queue for requests and executes them on a seperate thread, then updates the original request message with the generated image/s. Dante provides an example of how this would work with StableDiffusion, but should work with anything that takes in text and puts out an image if you are willing to build a module.
