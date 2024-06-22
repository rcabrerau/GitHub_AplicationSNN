import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import logging
import sys
import psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from codecarbon import EmissionsTracker
import snntorch as snn
from snntorch import spikegen

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_csnn.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])

#tracker = EmissionsTracker()
file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_16062024_0029.h5"

spiking_data_list = []
labels = []

label_map = {'center': 0, 'left': 1, 'right': 2}  # Define un mapeo de etiquetas a números

# Cargar datos desde el archivo HDF5
with h5py.File(file_path, 'r') as f:
    for group_name in f:
        if 'spiking_data' in f[group_name]:
            spiking_data = f[group_name]['spiking_data'][:]
            spiking_data_list.append(spiking_data)
            label = label_map[f[group_name].attrs['label']]
            labels.append(label)

# Convertir los datos a tensores de PyTorch
spiking_data_array = np.array(spiking_data_list, dtype=np.float32)
spiking_data_tensor = torch.tensor(spiking_data_array)
spiking_data_tensor_flat = spiking_data_tensor.view(-1, 3, 64, 64)
labels_flat = np.repeat(labels, 5)

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(spiking_data_tensor_flat, labels_flat, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Generar picos utilizando spikegen.latency
num_steps = 50  # Número de pasos de tiempo
spike_data = spikegen.latency(X_train, num_steps=num_steps)

# Imprimir los datos de picos generados
print(spike_data)
