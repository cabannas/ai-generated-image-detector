import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def smooth_curve(points, factor=0.4):
    """ Utiliza un factor de suavizado exponencial para suavizar la curva de la gráfica. """
    smoothed_points = np.zeros_like(points)
    smoothed_points[0] = points[0]
    for i in range(1, len(points)):
        smoothed_points[i] = smoothed_points[i - 1] * factor + points[i] * (1 - factor)
    return smoothed_points

def plot_metrics(model_path):
    model_name = os.path.basename(model_path)  # Extrae el nombre del modelo del path

    figures_path = f"{model_path}/figures"
    os.makedirs(figures_path, exist_ok=True)  # Crea el directorio de figuras si no existe

    # Construye las rutas completas a los archivos CSV
    path_accuracy = f'{model_path}/accuracy.csv'
    path_ap = f'{model_path}/ap.csv'
    path_loss = f'{model_path}/loss.csv'

    # Carga los datos
    data_accuracy = pd.read_csv(path_accuracy)
    data_ap = pd.read_csv(path_ap)
    data_loss = pd.read_csv(path_loss)

    # Suavizado de los datos con factores de suavizado diferentes
    smooth_accuracy = smooth_curve(data_accuracy['Value'], factor=0.4)
    smooth_ap = smooth_curve(data_ap['Value'], factor=0.4)
    smooth_loss = smooth_curve(data_loss['Value'], factor=0.6)  # Factor de suavizado aumentado para loss

    # Graficar la pérdida
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(data_loss['Step'], smooth_loss, label='Loss', color='tab:blue')
    ax_loss.set_title(f'Loss over Training Steps for {model_name}')
    ax_loss.set_xlabel('Training Steps')
    ax_loss.set_ylabel('Loss')
    ax_loss.grid(True)
    fig_loss.savefig(f'{figures_path}/Loss_over_Training_Steps.png')

    # Graficar la precisión
    fig_accuracy, ax_accuracy = plt.subplots()
    ax_accuracy.plot(data_accuracy['Step'], smooth_accuracy, label='Accuracy', color='tab:green')
    ax_accuracy.set_title(f'Accuracy over Training Steps for {model_name}')
    ax_accuracy.set_xlabel('Training Steps')
    ax_accuracy.set_ylabel('Accuracy')
    ax_accuracy.grid(True)
    fig_accuracy.savefig(f'{figures_path}/Accuracy_over_Training_Steps.png')

    # Graficar Average Precision
    fig_ap, ax_ap = plt.subplots()
    ax_ap.plot(data_ap['Step'], smooth_ap, label='AP', color='tab:red')
    ax_ap.set_title(f'Average Precision over Training Steps for {model_name}')
    ax_ap.set_xlabel('Training Steps')
    ax_ap.set_ylabel('Average Precision')
    ax_ap.grid(True)
    fig_ap.savefig(f'{figures_path}/Average_Precision_over_Training_Steps.png')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_folder = sys.argv[1]
        plot_metrics(model_folder)
    else:
        print("Please provide the model folder path.")
