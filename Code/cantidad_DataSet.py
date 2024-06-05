import h5py

# Abre el archivo HDF5 en modo lectura
with h5py.File("E:\MASTER UOC\AULAS_4TO_SEMESTRE\TFM\AplicationSNN\spiking_data_04062024_2344.h5", "r") as f:                          
    # Itera sobre los grupos del archivo
    total_datos = 0
    for group_name in f:
        # Obtén el tamaño de los datos del grupo actual
        datos_grupo = f[group_name]["spiking_data"].shape
        # Suma el tamaño de los datos del grupo actual al total
        total_datos += datos_grupo[0]  # El tamaño de los datos está en la primera dimensión

print("La cantidad total de datos en el archivo HDF5 es:", total_datos)
