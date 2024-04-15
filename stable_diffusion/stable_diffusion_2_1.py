import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


generator = torch.Generator(device='cuda').manual_seed(2)

prompt = "the most realistic portrait picture of a woman"

image = pipe(prompt, height=512, width=512, generator=generator).images[0]
    
image.save("stable_diffusion/woman.png")

image