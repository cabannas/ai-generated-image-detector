import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm 
import random
import os


# Definir listas de atributos con sus respectivos pesos
glasses_options = [('glasses', 30), ('no glasses', 70)]
eye_color_options = [('blue eyes', 10), ('green eyes', 10), ('brown eyes', 80)]
hair_length_options = [('curly hair', 20), ('short hair', 40), ('long hair', 40)]
freckles_options = [('freckles', 30), ('no freckles', 70)]
facial_expression_options = [('smiling', 50), ('serious', 50)]
gender_options = [('man', 50), ('woman', 50)]
background_options = [('office', 33), ('urban street', 33), ('park', 34)]

# Función para elegir un atributo basado en la ponderación
def weighted_choice(options):
    choices, weights = zip(*options)
    return random.choices(choices, weights=weights, k=1)[0]

# Generar un prompt con atributos ponderados
def create_prompt():
    glasses = weighted_choice(glasses_options)
    eye_color = weighted_choice(eye_color_options)
    hair_length = weighted_choice(hair_length_options)
    freckles = weighted_choice(freckles_options)
    facial_expression = weighted_choice(facial_expression_options)
    gender = weighted_choice(gender_options)
    background = weighted_choice(background_options)

    # Combina los atributos en un prompt
    prompt = f"a centered portrait of a {facial_expression} {gender} with {eye_color}, {hair_length}, {glasses} and {freckles}, in a {background}"
    return prompt



model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Create output directory
output_dir = "stable_diffusion/fake_dataset"
os.makedirs(output_dir, exist_ok=True)

# Generate and save images
num_images = 1000  # Total number of images you want to generate
for i in tqdm(range(num_images), desc="Generating images"):
    # Randomize the seed for each image
    seed = torch.seed()
    prompt = create_prompt()

    # Generate image
    generator = torch.Generator(device='cuda').manual_seed(seed)
    image = pipe(prompt, height=512, width=512, generator=generator).images[0]

    # Save image
    filename = f"{output_dir}/fi_{i+1001}.png"
    image.save(filename)
    print(f"Saved {filename}")

print("Dataset generation complete!")
