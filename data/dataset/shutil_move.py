import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def split_data(source_dir, target_dirs, label, ratios):
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(files)  # Mezclar aleatoriamente los archivos

    split_at = {
        'train': int(len(files) * ratios['train']),
        'val': int(len(files) * (ratios['train'] + ratios['val'])),
    }

    target_dirs['train'].mkdir(parents=True, exist_ok=True)
    target_dirs['val'].mkdir(parents=True, exist_ok=True)
    target_dirs['test'].mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(tqdm(files, desc=f"Procesando imágenes {label}")):
        if i < split_at['train']:
            shutil.copy(os.path.join(source_dir, file), target_dirs['train'] / label)
        elif i < split_at['val']:
            shutil.copy(os.path.join(source_dir, file), target_dirs['val'] / label)
        else:
            shutil.copy(os.path.join(source_dir, file), target_dirs['test'] / label)

# Rutas a los directorios
current_dir = Path(__file__).parent
dataset_dir = current_dir / 'formatted'

fake_dir = dataset_dir / 'fake_imgs'
real_dir = dataset_dir / 'real_imgs'

train_dir = current_dir / 'train'
val_dir = current_dir / 'val'
test_dir = current_dir / 'test'

# Crear directorios para 'train', 'val' y 'test'
os.makedirs(train_dir / '1_fake', exist_ok=True)
os.makedirs(train_dir / '0_real', exist_ok=True)
os.makedirs(val_dir / '1_fake', exist_ok=True)
os.makedirs(val_dir / '0_real', exist_ok=True)
os.makedirs(test_dir / '1_fake', exist_ok=True)
os.makedirs(test_dir / '0_real', exist_ok=True)

# Distribuir las imágenes
ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
split_data(fake_dir, {'train': train_dir, 'val': val_dir, 'test': test_dir}, '1_fake', ratios)
split_data(real_dir, {'train': train_dir, 'val': val_dir, 'test': test_dir}, '0_real', ratios)
