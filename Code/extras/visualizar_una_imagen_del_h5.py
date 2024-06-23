import h5py
import numpy as np
import matplotlib.pyplot as plt

# Ruta al archivo HDF5
file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_22062024_1529_test.h5"

# Función para cargar y mostrar una imagen
def load_and_show_image(file_path, image_index):
    with h5py.File(file_path, 'r') as f:
        # Construir el nombre del grupo
        group_name = f"image_{image_index}/spiking_data"
        
        # Cargar los datos de la imagen
        image_data = f[group_name][:]
        
        # Seleccionar la primera imagen del conjunto si tiene más de una dimensión adicional
        if image_data.ndim == 4:
            image_data = image_data[0]  # Seleccionamos la primera imagen
        
        # Si la imagen tiene un canal adicional, eliminarlo (por ejemplo, para datos RGB)
        if image_data.shape[0] == 3:
            image_data = np.transpose(image_data, (1, 2, 0))
        
        # Mostrar la imagen
        plt.imshow(image_data, cmap='gray')
        plt.title(f"Image {image_index}")
        plt.axis('off')
        plt.show()

# Cargar y mostrar la imagen con índice 996
load_and_show_image(file_path, 996)
