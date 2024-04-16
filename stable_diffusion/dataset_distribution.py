from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random

# Define the size and quality for formatting the images
size = (512, 512)
quality = 95

# Define ratios for splitting data
ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

def format_and_split_images(source_dir, target_dirs, label, size, quality, ratios):
    # Get all image files in the source directory
    images = [f for f in source_dir.glob('*') if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    # Shuffle images
    random.shuffle(images)

    # Determine split indices
    split_at = {
        'train': int(len(images) * ratios['train']),
        'val': int(len(images) * (ratios['train'] + ratios['val'])),
    }

    # Process and move images
    for i, image_path in enumerate(tqdm(images, desc=f"Processing {label} images")):
        # Open and format image
        img = Image.open(image_path)
        img = img.resize(size, Image.Resampling.LANCZOS)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Determine the target directory
        if i < split_at['train']:
            subdir = 'train'
        elif i < split_at['val']:
            subdir = 'val'
        else:
            subdir = 'test'

        # Create the output path
        output_path = target_dirs[subdir] / image_path.name
        # Ensure the target directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the formatted image
        img.save(output_path, quality=quality)

# Paths to directories
script_directory = Path().cwd()
stable_diffusion_directory = script_directory / 'stable_diffusion'
fake_dataset_directory = stable_diffusion_directory / 'fake_dataset'
real_dataset_directory = stable_diffusion_directory / 'flickr_faces_hq'
dataset_directory = script_directory / 'dataset'

# Create target directories
train_fake_dir = dataset_directory / 'train' / '1_fake'
train_real_dir = dataset_directory / 'train' / '0_real'
val_fake_dir = dataset_directory / 'val' / '1_fake'
val_real_dir = dataset_directory / 'val' / '0_real'
test_fake_dir = dataset_directory / 'test' / 'ldm' / '1_fake'
test_real_dir = dataset_directory / 'test' / 'ldm' / '0_real'


# Process and split fake images
format_and_split_images(fake_dataset_directory, {
    'train': train_fake_dir, 
    'val': val_fake_dir, 
    'test': test_fake_dir
}, '1_fake', size, quality, ratios)

# Process and split real images
format_and_split_images(real_dataset_directory, {
    'train': train_real_dir, 
    'val': val_real_dir, 
    'test': test_real_dir
}, '0_real', size, quality, ratios)

print("Image formatting and distribution complete.")
