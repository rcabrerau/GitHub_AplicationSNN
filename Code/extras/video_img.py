import cv2
import os

# Directorio de imágenes
image_folder = 'E:\MASTER UOC\AULAS_4TO_SEMESTRE\TFM\AplicationSNN\datasetConvertion_CNN_to_SNN\IMG'
# Nombre del archivo de video de salida
video_name = 'video_salida.avi'

# Obtener las imágenes del directorio
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()  # Asegurar el orden correcto de las imágenes

# Leer la primera imagen para obtener el tamaño del video
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Definir el codec y crear el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Liberar el objeto VideoWriter
video.release()
cv2.destroyAllWindows()
