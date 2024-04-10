from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm

def format_images(input_path, output_path, size=(1024, 1024), quality=95):
    img = Image.open(input_path)
    img = img.resize(size, Image.Resampling.LANCZOS)

    # Convert to RGB (if necessary)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img.save(output_path, quality=quality)

# Obtén el directorio del script y los directorios de imágenes falsas y reales
script_directory = Path(__file__).parent
fake_images_directory = script_directory / 'fake_imgs'
real_images_directory = script_directory / 'real_imgs'

# Directorios de salida
output_directory_fake = script_directory / 'formatted' / 'fake_imgs'
output_directory_real = script_directory / 'formatted' / 'real_imgs'
output_directory_fake.mkdir(parents=True, exist_ok=True)
output_directory_real.mkdir(parents=True, exist_ok=True)

# Formatea las imágenes falsas
fake_images = [file for file in os.listdir(fake_images_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
for img_file in tqdm(fake_images, desc="Formatting fake images"):
    img_path = fake_images_directory / img_file
    output_img_path = output_directory_fake / img_file
    format_images(str(img_path), str(output_img_path))

# Formatea las imágenes reales
real_images = [file for file in os.listdir(real_images_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
for img_file in tqdm(real_images, desc="Formatting real images"):
    img_path = real_images_directory / img_file
    output_img_path = output_directory_real / img_file
    format_images(str(img_path), str(output_img_path))


