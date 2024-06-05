import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from codecarbon import EmissionsTracker
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

experiment = Experiment(
  api_key="tuUec82jIwUFw6SpIaMEIeCPb",
  project_name="ssn-emissions",
  workspace="rcabreraur-uoc-edu"
)

# Inicializar el rastreador de emisiones
tracker = EmissionsTracker()

# Ruta al archivo H5
file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/spiking_data.h5"

# Lista para almacenar los datos de espiking
spiking_data_list = []
labels = []

# Abrir el archivo H5 en modo lectura
with h5py.File(file_path, 'r') as f:
    # Iterar sobre cada grupo en el archivo H5
    for group_name in f:
        # Obtener el conjunto de datos 'spiking_data' del grupo actual
        spiking_data = f[group_name]['spiking_data'][:]
        # Agregar los datos a la lista
        spiking_data_list.append(spiking_data)
        # Obtener la etiqueta del grupo actual
        label = int(group_name.split('_')[1])  # Obtener el número de la etiqueta del grupo
        labels.append(label)

# Convertir la lista de datos a un solo numpy array
spiking_data_array = np.array(spiking_data_list, dtype=np.float32)

# Convertir el array numpy a un tensor de PyTorch
spiking_data_tensor = torch.tensor(spiking_data_array)

# Verificar la forma del tensor resultante
print("Forma del tensor de datos de espiking:", spiking_data_tensor.shape)

# Aplanar el tensor para obtener todas las imágenes sin agrupar
spiking_data_tensor_flat = spiking_data_tensor.view(-1, 3, 64, 64)

# Verificar la forma del tensor aplanado
print("Forma del tensor de datos de espiking (aplanado):", spiking_data_tensor_flat.shape)

# Replicar las etiquetas para cada imagen dentro de cada grupo
labels_flat = np.repeat(labels, 5)

# Separar el conjunto de datos en conjuntos de entrenamiento, prueba y validación
X_train, X_temp, y_train, y_temp = train_test_split(spiking_data_tensor_flat, labels_flat, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Verificar las formas de los conjuntos de datos y etiquetas
print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de y_test:", y_test.shape)
print("Forma de X_val:", X_val.shape)
print("Forma de y_val:", y_val.shape)

# Definir el modelo de red neuronal
class CNN_Spiking(nn.Module):
    def __init__(self):
        super(CNN_Spiking, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)  # Capa de salida para la tarea de regresión

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Aplanar los mapas de características
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Inicializar el modelo
model = CNN_Spiking()

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
num_epochs = 10          # incluir el numero de epocas más adecuado
batch_size = 32
n_samples = X_train.shape[0]
n_batches = n_samples // batch_size

# Iniciar el seguimiento de emisiones para el entrenamiento
#with tracker.emissions():
with tracker:       
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in tqdm(range(n_batches)):
            optimizer.zero_grad()  # Reiniciar los gradientes
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_X = X_train[start_idx:end_idx]  # Obtener el lote de datos de entrada
            batch_y = y_train[start_idx:end_idx]  # Obtener el lote de etiquetas
            
            # Pasar los datos reformateados al modelo
            outputs = model(batch_X)

            # Calcular la pérdida y realizar la retropropagación
            loss = criterion(outputs.squeeze(), torch.tensor(batch_y, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / (n_batches * batch_X.shape[1])}")

# Evaluar el modelo en el conjunto de validación
with torch.no_grad():
    #y_pred_val = model(X_val.unsqueeze(1)).squeeze()
    y_pred_val = model(X_val).squeeze()

# Calcular métricas en el conjunto de validación
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

print("Métricas en el conjunto de validación:")
print(f"MSE: {mse_val}")
print(f"MAE: {mae_val}")
print(f"R^2: {r2_val}")


# Report multiple hyperparameters using a dictionary:
hyper_params = {
   "learning_rate": 0.5,
   "steps": 100000,
   "batch_size": 32,
}
experiment.log_parameters(hyper_params)

# Initialize and train your model
# model = TheModelClass()
# train(model)

# Seamlessly log your Pytorch model
log_model(experiment, model=model, model_name="TheModel")














#-------eliminar todo lo que sigue

import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from codecarbon import EmissionsTracker
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

experiment = Experiment(
  api_key="tsffusfsfUecsdfs824234fsf234CPsfsdfb",
  project_name="ssn-emissions",
  workspace="rcabreraur-uoc-edu"
)







# Inicializar el rastreador de emisiones



tracker = EmissionsTracker()




file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/spiking_data.h5"
spiking_data_list = []
labels = []
with h5py.File(file_path, 'r') as f:
    for group_name in f:
        spiking_data = f[group_name]['spiking_data'][:]
        spiking_data_list.append(spiking_data)
        label = int(group_name.split('_')[1])  
        labels.append(label)
spiking_data_array = np.array(spiking_data_list, dtype=np.float32)
spiking_data_tensor = torch.tensor(spiking_data_array)






spiking_data_tensor_flat = spiking_data_tensor.view(-1, 3, 64, 64)
labels_flat = np.repeat(labels, 5)
X_train, X_temp, y_train, y_temp = train_test_split(spiking_data_tensor_flat, 
                                                    labels_flat, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)





class CNN_Spiking(nn.Module):
    def __init__(self):
        super(CNN_Spiking, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)  

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = CNN_Spiking()




criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)






num_epochs = 10        
batch_size = 32
n_samples = X_train.shape[0]
n_batches = n_samples // batch_size
with tracker:       
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in tqdm(range(n_batches)):
            optimizer.zero_grad()  
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_X = X_train[start_idx:end_idx]  
            batch_y = y_train[start_idx:end_idx] 
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), torch.tensor(batch_y, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / (n_batches * batch_X.shape[1])}")





with torch.no_grad():
    y_pred_val = model(X_val).squeeze()
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)
print("Métricas en el conjunto de validación:")
print(f"MSE: {mse_val}")
print(f"MAE: {mae_val}")
print(f"R^2: {r2_val}")






hyper_params = {
   "learning_rate": 0.5,
   "steps": 100000,
   "batch_size": 32,
}
experiment.log_parameters(hyper_params)
log_model(experiment, model=model, model_name="TheModel")

