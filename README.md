El presente repositorio compende los códigos de implementación requeridos en el estudio de investigación del Trabajo de Fin de Master en Ciencia de Datos de la Universitat Oberta de Catalunya.
TFM: Evaluación de la Ecoeficiencia en Redes Neuronales Spiking para Conducción Autónoma
Autor: Robinson Xavier Cabrera Ureña (<rcabreraur@uoc.edu>)
Tutor: Raul Parada Medina (<rparada@uoc.edu>)

<br>
conversion_to_spiking_dataset_snnTorch.py: Código de entrenamiento del set de datos h5 (datos convencionales convertios a tipos de datos spiking).

cnn_spiking.py: Código principal en el cual se basa el TFM corresponden a la conversión de un dataset y un entrenamiento de una red neuronal convolucional spiking.

cnn.py: Código de implementación de un modelo de red neuronal convolucional.

Como apoyo en el estudio se ha desarrollado otros archivos que han permitido identificar la cantidad de registros, el uso de la GPU, la visualización de las imágenes y una perspectiva en formato de video de las imágenes almacenadas:
cantidad_DataSet.py: Código implementado para conocer la cantidad de registros en el dataset h5.

consultar_GPU_Laptop.py: Código implementado para saber si el equipo dispone de una GPU.

video_img.py: Código implementado para unificar las imágenes del set de datos y permitir una perspectiva en formato de video.

visualizar_archivoH5_v2.py: Código implementado para visualizar los archivos almacenados en el formato h5.
