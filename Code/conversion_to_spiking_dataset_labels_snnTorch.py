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
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_convert_spiking.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])

# Carpetas para cada categoría
image_folders = {
    'center': 'E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/IMG/center',
    'left': 'E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/IMG/left',
    'right': 'E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/IMG/right'
}

time_window = 100  # Ventana de tiempo en milisegundos
threshold = 0.001  # Umbral para la generación de eventos de espiking 

tracker = EmissionsTracker() 

class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files                
        self.transform = transform                   
    def __len__(self):                               
        return len(self.image_files)                 
    def __getitem__(self, idx):                       
        image_path, label = self.image_files[idx]
        image = io.imread(image_path)
        image = Image.fromarray(image)               
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
        return image, label, transform_info                  

preprocess_transform = transforms.Compose([
    # Aumento de imagenes
    RandomRotation(degrees=30),                 
    RandomHorizontalFlip(p=0.5),                
    ColorJitter(brightness=0.2, contrast=0.2),  
    # Transformación
    transforms.Resize((64, 64)),                
    transforms.ToTensor(),                      
])

image_files = []
for label, folder in image_folders.items():
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)):
            image_files.append((os.path.join(folder, f), label))
logging.info(f"Total images found: {len(image_files)}")

def show_transformed_images(image_batch):
    fig, axes = plt.subplots(2, len(image_batch), figsize=(15, 5))
    for idx, (image_path, label) in enumerate(image_batch):
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
        axes[0, idx].set_title(f"Original ({label})")
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(transformed_image_pil)
        axes[1, idx].set_title("Transformada")
        axes[1, idx].axis('off')
        
        axes[1, idx].text(0.5, -0.1, "\n".join(transform_info), 
                          ha='center', va='top', transform=axes[1, idx].transAxes, fontsize=9)

    plt.tight_layout()
    plt.show()

batch_size = 1000
image_batches = [image_files[i:i+batch_size] for i in range(0, len(image_files), batch_size)]

def process_image_batch(image_batch):           
    import snntorch
    # Preprocesar el lote de imágenes
    image_dataset = ImageDataset(image_batch, transform=preprocess_transform)
    spiking_data_batch = []
    labels_batch = []
    for image, label, transform_info in image_dataset:
        # Convertir el lote de imágenes a eventos de espiking
        spiking_data = convert_to_spikes(image, time_window, threshold)
        spiking_data_batch.append(spiking_data)
        labels_batch.append(label)
    logging.info(f"Processed batch of size: {len(image_batch)}")  
    return spiking_data_batch, labels_batch                   

def convert_to_spikes(image, time_window, threshold):       
    # Realizar la conversión utilizando snntorch    
    spiking_data = spikegen.latency(image, num_steps=5, normalize=True, linear=True)    
    return spiking_data                         

if __name__ == '__main__':
    show_transformed_images(image_files[:5])
    
    # Procesar los lotes de imágenes utilizando multiprocessing
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        with tracker:
            with h5py.File("spiking_data.h5", "w") as f:
                for i, (spiking_data_batch, labels_batch) in enumerate(tqdm(pool.imap(process_image_batch, image_batches), total=len(image_batches))):
                    for j, (spiking_data, label) in enumerate(zip(spiking_data_batch, labels_batch)):
                        group = f.create_group(f"image_{i * batch_size + j}")
                        group.create_dataset("spiking_data", data=spiking_data)
                        group.attrs['label'] = label  # Guardar la etiqueta como atributo
                    logging.info(f"Saved batch {i+1} to H5 file")
    logging.info(f"Total batches processed: {len(image_files)}")
