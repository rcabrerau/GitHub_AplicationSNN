import os
from torch.utils.data import Dataset
from torchvision import transforms
import multiprocessing
import h5py
from tqdm import tqdm
from skimage import io
from snntorch import spikegen
from torchvision.transforms import ColorJitter, RandomRotation, RandomHorizontalFlip
from PIL import Image
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import random
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model


experiment = Experiment(
  api_key="tuUec82jIwUFw6SpIaMEIeCPb",
  project_name="ssn-conversion",
  workspace="rcabreraur-uoc-edu"
)


# Carpeta que contiene las imágenes
image_folder = 'E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/IMG'

# Parámetros para la conversión a eventos de espiking
time_window = 100  # Ventana de tiempo en milisegundos
threshold = 0.001  # Umbral para la generación de eventos de espiking 

# Inicializar el tracker de emisiones
tracker = EmissionsTracker() 

# Clase Dataset para cargar y preprocesar las imágenes, con aumento de datos
class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files                # image_files (list): Lista de rutas de archivos de imágenes.
        self.transform = transform                    # Transformación a aplicar a las imágenes. Por defecto, None.

    def __len__(self):                                # Retorna el número total de imágenes en el dataset.
        return len(self.image_files)                  # Número total de imágenes.

    def __getitem__(self, idx):                       # idx (int): Índice de la imagen.
        image_path = self.image_files[idx]
        image = io.imread(image_path)
        image = Image.fromarray(image)                # Convertir a formato PIL
        transform_info = []
        if self.transform:
            for t in self.transform.transforms:
                if isinstance(t, RandomRotation):
                    degrees = t.get_params(t.degrees)
                    transform_info.append(f'Rotación: {degrees} grados')
                elif isinstance(t, RandomHorizontalFlip):
                    if random.random() < t.p:
                        image = t(image)
                        transform_info.append('Volteo horizontal')
                    continue
                elif isinstance(t, ColorJitter):
                    transform_info.append('Cambio en brillo y contraste')
                elif isinstance(t, transforms.Resize):
                    transform_info.append('Redimensionado a 64x64')
                image = t(image)
        return image, transform_info                  # Retorna una imagen del dataset en la posición especificada por el índice.

# Transformación para el preprocesamiento de imágenes
preprocess_transform = transforms.Compose([
    # Aumento de imagenes
    RandomRotation(degrees=30),                 # Rotación aleatoria en un rango de -30 a 30 grados
    RandomHorizontalFlip(p=0.5),                # Volteo horizontal aleatorio con una probabilidad de 0.5
    ColorJitter(brightness=0.2, contrast=0.2),  # Cambio aleatorio en brillo y contraste
    # Transformación
    transforms.Resize((64, 64)),                # Ajustar al tamaño deseado
    transforms.ToTensor(),                      # Convertir a tensor
])

# Obtener la lista de archivos de imágenes
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
print(f"Total images found: {len(image_files)}")

# Función para mostrar imágenes originales y transformadas
def show_transformed_images(image_batch):
    fig, axes = plt.subplots(2, len(image_batch), figsize=(15, 5))
    for idx, image_path in enumerate(image_batch):
        original_image = Image.open(image_path)
        image = original_image.copy()
        transform_info = []
        for t in preprocess_transform.transforms:
            if isinstance(t, RandomRotation):
                degrees = t.get_params(t.degrees)
                transform_info.append(f'Rotación: {degrees:.1f} grados')
                image = t(image)
            elif isinstance(t, RandomHorizontalFlip):
                if random.random() < t.p:
                    image = t(image)
                    transform_info.append('Volteo horizontal')
                continue
            elif isinstance(t, ColorJitter):
                transform_info.append('Cambio en brillo y contraste')
                image = t(image)
            elif isinstance(t, transforms.Resize):
                transform_info.append('Redimensionado a 64x64')
                image = t(image)
        
        transformed_image_pil = transforms.ToPILImage()(transforms.ToTensor()(image))
        
        axes[0, idx].imshow(original_image)
        axes[0, idx].set_title("Original")
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(transformed_image_pil)
        axes[1, idx].set_title("Transformada")#\n" + "\n".join(transform_info))
        axes[1, idx].axis('off')
        
        
        # Añadir la información de las transformaciones debajo de la imagen transformada
        axes[1, idx].text(0.5, -0.1, "\n".join(transform_info), 
                          ha='center', va='top', transform=axes[1, idx].transAxes, fontsize=9)

    plt.tight_layout()
    plt.show()

# Dividir la lista de archivos en lotes más pequeños
batch_size = 1000
image_batches = [image_files[i:i+batch_size] for i in range(0, len(image_files), batch_size)]

# Función para procesar un lote de imágenes
def process_image_batch(image_batch):           # image_batch (list): Lote de rutas de archivos de imágenes.
    import snntorch
    # Preprocesar el lote de imágenes
    image_dataset = ImageDataset(image_batch, transform=preprocess_transform)
    spiking_data_batch = []
    for image, transform_info in image_dataset:
        # Convertir el lote de imágenes a eventos de espiking
        spiking_data = convert_to_spikes(image, time_window, threshold)
        spiking_data_batch.append(spiking_data)
    print(f"Processed batch of size: {len(image_batch)}")  # Añadir registro
    return spiking_data_batch                   # Lista de datos de espiking procesados para cada imagen en el lote.

# Función para convertir imágenes a eventos de espiking con codificación de latencia
# image (torch.Tensor): Imagen en formato tensor.
# time_window (int): Ventana de tiempo en milisegundos.
# threshold (float): Umbral para la generación de eventos de espiking.
def convert_to_spikes(image, time_window, threshold):       
    # Realizar la conversión utilizando snntorch    
    spiking_data = spikegen.latency(image, num_steps=5, normalize=True, linear=True)    
    return spiking_data                         # Datos de espiking generados a partir de la imagen.

if __name__ == '__main__':
    # Mostrar un lote pequeño de imágenes transformadas
    show_transformed_images(image_files[:5])
    
    # Procesar los lotes de imágenes utilizando multiprocessing
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Utilizar tqdm para seguir el progreso del procesamiento de lotes
        with tracker:
            spiking_data_list = []
            with h5py.File("spiking_data.h5", "w") as f:
                for i, spiking_data_batch in enumerate(tqdm(pool.imap(process_image_batch, image_batches), total=len(image_batches))):
                    for j, spiking_data in enumerate(spiking_data_batch):
                        group = f.create_group(f"image_{i * batch_size + j}")
                        group.create_dataset("spiking_data", data=spiking_data)
                    print(f"Saved batch {i+1} to H5 file")  # Añadir registro
    print(f"Total batches processed: {len(spiking_data_list)}")
