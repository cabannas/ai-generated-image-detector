import os
import requests
from PIL import Image
from io import BytesIO
import csv

# Ruta al archivo CSV
csv_file_path = 'data/dataset_images_diffusion.csv'

# Ruta de la carpeta donde se guardarán las imágenes procesadas
output_folder_path = 'data/dataset/fake'

# Crear la carpeta si no existe
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Función para descargar y guardar una imagen
def download_and_process_image(image_url, output_folder):
    try:
        # Obtener el nombre de la imagen del URL para usarlo al guardar la imagen procesada
        image_name = os.path.basename(image_url)
        response = requests.get(image_url)
        response.raise_for_status()  # Verifica que la solicitud fue exitosa

        # Abre la imagen de la respuesta HTTP y recorta los últimos 50 píxeles
        with Image.open(BytesIO(response.content)) as img:
            cropped_img = img.crop((0, 0, img.width, img.height - 50))
        
        # Ruta de salida para la imagen recortada
        output_path = os.path.join(output_folder, image_name)

        # Guarda la imagen recortada
        cropped_img.save(output_path)
        print(f"Imagen guardada en: {output_path}")

    except requests.exceptions.HTTPError as http_err:
        print(f"Error HTTP: {http_err} - URL: {image_url}")
    except Exception as err:
        print(f"Error: {err} - URL: {image_url}")

# Leer los URLs del archivo CSV y procesar cada imagen
with open(csv_file_path, newline='') as csvfile:
    image_urls = csv.reader(csvfile)
    for row in image_urls:
        if row:  # Asegurarse de que la fila no esté vacía
            download_and_process_image(row[0], output_folder_path)
