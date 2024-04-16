import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm 
import random
import os

''' PARAMS '''

num_images = 600 # Total number of images you want to generate
num_rename = 10000 # Number in which naming starts

# Define atributes and weights

race_options = [('asian', 20), ('african', 20), ('hindu', 20), ('caucasian', 40)]
glasses_options = [('glasses', 30), ('no glasses', 70)]
eye_color_options = [('blue eyes', 10), ('green eyes', 10), ('brown eyes', 80)]
hair_length_options = [('curly hair', 15), ('short hair', 45), ('long hair', 40)]
freckles_options = [('freckles', 20), ('no freckles', 80)]
facial_expression_options = [('smiling', 50), ('serious', 50)]
age_n_gender_options = [('baby boy', 10), ('baby girl', 10),
                        ('young boy', 10), ('young girl', 10),
                        ('teenager boy', 10), ('teenager girl', 10),
                        ('man', 15), ('woman', 15),
                        ('old man', 5), ('old woman', 5)]
background_options = [('office', 33), ('urban street', 33), ('park', 34)]

''' END PARAMS '''



def weighted_choice(options):
    choices, weights = zip(*options)
    return random.choices(choices, weights=weights, k=1)[0]

def create_prompt():
    race = weighted_choice(race_options)
    glasses = weighted_choice(glasses_options)
    eye_color = weighted_choice(eye_color_options)
    hair_length = weighted_choice(hair_length_options)
    freckles = weighted_choice(freckles_options)
    facial_expression = weighted_choice(facial_expression_options)
    age_n_gender = weighted_choice(age_n_gender_options)
    background = weighted_choice(background_options)

    # Modify prompt at will
    prompt = f"a centered portrait of a {facial_expression} {race} {age_n_gender} with {eye_color}, {hair_length}, {glasses} and {freckles}, in a {background}"
    return prompt



model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Create output directory
output_dir = "stable_diffusion/fake_dataset"
os.makedirs(output_dir, exist_ok=True)

# Generate and save images
for i in tqdm(range(num_images), desc="Generating images"):
    # Randomize the seed for each image
    seed = torch.seed()
    prompt = create_prompt()

    # Generate image
    generator = torch.Generator(device='cuda').manual_seed(seed)
    image = pipe(prompt, height=512, width=512, generator=generator).images[0]

    # Save image
    filename = f"{output_dir}/fi_{i+num_rename}.png"
    image.save(filename)
    print(f"Saved {filename}")

print("Dataset generation complete!")
