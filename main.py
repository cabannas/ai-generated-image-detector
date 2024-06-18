import torch
from torchvision import transforms
from PIL import Image
from networks.resnet import resnet50

# Verifica si CUDA (GPU) está disponible y configura el dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path):
    # Carga el modelo preentrenado de ResNet50 con una salida binaria
    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location=device)
    # El estado del modelo puede necesitar ser modificado ligeramente dependiendo de si se usó 'DataParallel' durante el entrenamiento
    model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
    model.to(device)  # Transfiere el modelo al dispositivo (GPU/CPU)
    model.eval()  # Coloca el modelo en modo de evaluación
    return model

def preprocess_image(image_path, crop_size=224):
    # Transforma la imagen para el modelo
    trans = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return trans(image).unsqueeze(0)  # Añade una dimensión para batch_size=1

def predict(model, img_tensor):
    img_tensor = img_tensor.to(device)  # Transfiere los datos de entrada al dispositivo
    # Realiza la predicción usando el modelo
    with torch.no_grad():
        prediction = model(img_tensor)
        # Aplica sigmoid para obtener la probabilidad de la clase 'fake'
        probability = prediction.sigmoid().item()
    return probability

# Configura las rutas al modelo y a la imagen que deseas evaluar
model_path = 'weights/every50.pth'  # Asegúrate de usar la ruta correcta
image_path = 'examples/fakegpt.png'

# Carga el modelo y la imagen
model = load_model(model_path)
img_tensor = preprocess_image(image_path)

# Obtiene la probabilidad de que la imagen sea sintética
probability_fake = predict(model, img_tensor)
print(f'Probability of being synthetic: {probability_fake * 100:.2f}%')
