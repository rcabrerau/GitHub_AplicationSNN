import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from codecarbon import EmissionsTracker
import logging
import sys
import psutil

# Función para imprimir el uso de memoria
def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

# Configurar el logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])

# Inicializar el rastreador de emisiones
tracker = EmissionsTracker()

# Ruta al archivo H5
file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/spiking_data.h5"

# Lista para almacenar los datos de espiking
spiking_data_list = []
labels = []

# Abrir el archivo H5 en modo lectura
with h5py.File(file_path, 'r') as f:
    # Seleccionar un subconjunto de 10,000 registros
    count = 0
    for group_name in f:
        if count >= 10000:
            break
        spiking_data = f[group_name]['spiking_data'][:]
        spiking_data_list.append(spiking_data)
        label = int(group_name.split('_')[1])
        labels.append(label)
        count += 1

# Convertir la lista en un array numpy
spiking_data_array = np.array(spiking_data_list, dtype=np.float32)

# Normalización de los datos de espiking por lotes con barra de progreso
def normalize_in_batches(data, batch_size=1000):
    data_normalized = np.empty_like(data)
    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    for i in tqdm(range(0, len(data), batch_size), desc="Normalizing Batches"):
        batch = data[i:i + batch_size]
        mean = np.mean(batch, axis=0)
        std = np.std(batch, axis=0)

        # Evitar la división por cero
        std[std == 0] = 1e-8

        data_normalized[i:i + batch_size] = (batch - mean) / std

        # Verificar si hay NaN en el lote normalizado
        if np.isnan(data_normalized[i:i + batch_size]).any():
            logging.error(f"NaN encontrado en el lote {i // batch_size + 1}/{num_batches}")
            raise ValueError("NaN encontrado durante la normalización de datos")

        logging.info(f"Lote {i // batch_size + 1}/{num_batches} normalizado")
    return data_normalized

# Normalizar los datos
spiking_data_array = normalize_in_batches(spiking_data_array)

# Convertir los datos normalizados en un tensor de PyTorch
spiking_data_tensor = torch.tensor(spiking_data_array)

# Aplanar los datos para que se ajusten al modelo
spiking_data_tensor_flat = spiking_data_tensor.view(-1, 3, 64, 64)
labels_flat = np.repeat(labels, 5)

# Dividir los datos en conjuntos de entrenamiento, prueba y validación
X_train, X_temp, y_train, y_temp = train_test_split(spiking_data_tensor_flat, labels_flat, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Verificar si hay NaN en los datos de entrenamiento y validación
if np.isnan(X_train.numpy()).any() or np.isnan(X_val.numpy()).any():
    raise ValueError("Datos de entrenamiento o validación contienen NaN")

# Definir la arquitectura del modelo
class DeepCSNN(nn.Module):
    def __init__(self):
        super(DeepCSNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Inicializar el modelo, la función de pérdida y el optimizador
model_csnn = DeepCSNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_csnn.parameters(), lr=0.0001)

# Entrenamiento del modelo por lotes con barra de progreso
num_epochs = 1
batch_size = 16

# Función para entrenar el modelo por lotes
def train_model_in_batches(model, criterion, optimizer, X_train, y_train, num_epochs, batch_size):
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        with tqdm(total=n_batches, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), torch.tensor(batch_y, dtype=torch.float32))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"Loss": epoch_loss / (i + 1)})
        logging.info(f"Epoch {epoch + 1}, Loss: {epoch_loss / n_batches}")
        print_memory_usage()

# Iniciar el rastreador de emisiones y entrenar el modelo
with tracker:
    train_model_in_batches(model_csnn, criterion, optimizer, X_train, y_train, num_epochs, batch_size)

# Función para predecir en lotes con barra de progreso
def predict_in_batches(model, data, batch_size):
    predictions = []
    n_samples = data.shape[0]
    n_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)
    with torch.no_grad():
        with tqdm(total=n_batches, desc="Predicting Batches") as pbar:
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_data = data[start_idx:end_idx]
                batch_preds = model(batch_data).squeeze().cpu().numpy()
                predictions.extend(batch_preds)
                pbar.update(1)
    return np.array(predictions)

# Evaluar el modelo en el conjunto de validación
model_csnn.eval()
y_pred_val = predict_in_batches(model_csnn, X_val, batch_size=16)

# Verificar si hay NaN en las predicciones o en las etiquetas
if np.isnan(y_pred_val).any() or np.isnan(y_val).any():
    raise ValueError("Predicciones o etiquetas de validación contienen NaN")

# Calcular métricas de rendimiento
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

logging.info("Métricas en el conjunto de validación (CSNN):")
logging.info(f"MSE: {mse_val}")
logging.info(f"MAE: {mae_val}")
logging.info(f"R^2: {r2_val}")
