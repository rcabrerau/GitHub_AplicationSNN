import os
from PIL import Image
from torchvision import transforms
from codecarbon import EmissionsTracker
import logging
import sys

# Configurar el logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_aumentodatos_setJPG.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])

# Configuración de rutas
data_dir = "E:\MASTER UOC\AULAS_4TO_SEMESTRE\TFM\AplicationSNN\datasetConvertion_CNN_to_SNN\IMG"
output_dir = "E:\MASTER UOC\AULAS_4TO_SEMESTRE\TFM\AplicationSNN\datasetConvertion_CNN_to_SNN\IMG_AUGMENTED"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tracker = EmissionsTracker()

# Transformaciones de imagen
augment_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# Función para guardar imágenes aumentadas
def save_augmented_images(image_path, output_folder, label, augment_transform, num_augmentations=5):
    original_image = Image.open(image_path)
    for i in range(num_augmentations):
        augmented_image = original_image.copy()
        augmented_image = augment_transform(augmented_image)
        augmented_image.save(os.path.join(output_folder, f"{label}_aug_{i}_{os.path.basename(image_path)}"))

# Crear imágenes aumentadas
with tracker:
    image_folders = {'Center': 0, 'Right': 1, 'Left': 2}
    for label in image_folders.keys():
        folder_path = os.path.join(data_dir, label)
        output_folder = os.path.join(output_dir, label)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for f in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, f)):
                save_augmented_images(os.path.join(folder_path, f), output_folder, label, augment_transform)

# Imprimir la cantidad de imágenes en cada carpeta
for label in image_folders.keys():
    output_folder = os.path.join(output_dir, label)
    num_images = len([name for name in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, name))])
    logging.info(f"Carpeta '{label}': {num_images} imágenes guardadas.")

logging.info("Conjunto de datos aumentado creado y guardado correctamente.")
