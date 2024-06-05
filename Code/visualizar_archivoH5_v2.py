import h5py
import matplotlib.pyplot as plt


# Abre el archivo HDF5 en modo lectura
""" with h5py.File("spiking_data.h5", "r") as f:
    # Itera sobre los grupos en el archivo
    for group_name in f:
        print("Grupo:", group_name)
        # Accede al grupo actual
        group = f[group_name]
        # Itera sobre los conjuntos de datos en el grupo
        for dataset_name in group:
            print("    Conjunto de datos:", dataset_name)
            # Accede al conjunto de datos actual
            dataset = group[dataset_name]
            # Imprime la forma del conjunto de datos
            print("        Forma:", dataset.shape) """



# Abre el archivo HDF5 en modo lectura
with h5py.File("E:\MASTER UOC\AULAS_4TO_SEMESTRE\TFM\AplicationSNN\spiking_data_20052024_1946.h5", "r") as f:
    # Recorre todos los grupos del archivo
    for group_name in f:
        # Obtiene los datos de espiking del grupo actual
        spiking_data = f[group_name]["spiking_data"][:]
        
        # Visualiza los datos de espiking para cada representación temporal
        for i, temporal_data in enumerate(spiking_data):
            plt.figure()
            # Transpone la imagen para que los canales estén en la última dimensión
            temporal_data = temporal_data.transpose(1, 2, 0)
            plt.imshow(temporal_data)
            plt.title(f"Spiking Data - {group_name} - Temporal Representation {i}")
            plt.xlabel("Width")
            plt.ylabel("Height")
            plt.colorbar(label="Spike Intensity")
            plt.show()
