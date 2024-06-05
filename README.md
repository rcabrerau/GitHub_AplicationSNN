El código principal en el cual se basa el TFM corresponden a la conversión de un dataset y un entrenamiento de una red neuronal convolucional spiking:
cnn_spiking_v5.py
Código de entrenamiento del set de datos h5 (datos convencionales convertios a tipos de datos spiking)

conversion_to_spiking_dataset_snnTorch_v6.py
Código implementado para la conversión de un dataset convensional de archivos jpg a un set de datos spiking.


Como apoyo en el estudio se ha desarrollado otros archivos que han permitido identificar la cantidad de registros, el uso de la GPU, la visualización de las imágenes  y una perspectiva en formato de video de las imágenes almacenadas:
cantidad_DataSet.py
Código implementado para conocer la cantidad de registros en el dataset h5.

consultar_GPU_Laptop.py
Código implementado para saber si el equipo dispone de una GPU.

video_img.py
Código implementado para unificar las imágenes del set de datos y permitir una perspectiva en formato de video.

visualizar_archivoH5_v2.py
Código implementado para visualizar los archivos almacenados en el formato h5.
