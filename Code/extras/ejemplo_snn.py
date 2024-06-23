
import torch
from sklearn.model_selection import train_test_split
import snntorch as snn
from snntorch import spikegen
import h5py
import numpy as np

# Ruta al archivo HDF5
file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_22062024_1529_test.h5"

spiking_data_list = []
labels = []

label_map = {'center': 0, 'left': 1, 'right': 2}  # Define un mapeo de etiquetas a números

# Función para cargar datos desde el archivo HDF5
def load_data(file_path):
    spiking_data_list = []
    labels = []

    with h5py.File(file_path, 'r') as f:
        for group_name in f:
            if 'spiking_data' in f[group_name]:
                spiking_data = f[group_name]['spiking_data'][:]
                spiking_data_list.append(spiking_data)
                label = label_map[f[group_name].attrs['label']]  # Usa el mapeo para convertir la etiqueta a un número
                labels.append(label)

    spiking_data_array = np.array(spiking_data_list, dtype=np.float32)
    spiking_data_tensor = torch.tensor(spiking_data_array)
    spiking_data_tensor_flat = spiking_data_tensor.view(-1, 3, 64, 64)
    labels_flat = np.repeat(labels, 5)

    X_train, X_temp, y_train, y_temp = train_test_split(spiking_data_tensor_flat, labels_flat, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val

# Cargar los datos
X_train, y_train, X_val, y_val = load_data(file_path)

# Seleccionar la primera imagen del conjunto si tiene más de una dimensión adicional
# Supongamos que queremos usar la primera imagen del primer grupo para simplicidad
if X_train.ndim == 4:
    X_train = X_train[0, 0]

# Definir el número de pasos de tiempo
num_steps = 5

# Generar picos usando codificación de latencia con normalización
spike_data = spikegen.latency(X_train, num_steps=num_steps, normalize=True)

# Imprimir los datos de picos generados
print(spike_data)


