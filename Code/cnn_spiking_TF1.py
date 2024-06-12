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
    for group_name in f:
        spiking_data = f[group_name]['spiking_data'][:]
        spiking_data_list.append(spiking_data)
        label = int(group_name.split('_')[1])
        labels.append(label)

spiking_data_array = np.array(spiking_data_list, dtype=np.float32)

# Normalización de los datos de espiking por lotes
""" def normalize_in_batches(data, batch_size=1000):
    data_normalized = np.empty_like(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        mean = np.mean(batch, axis=0)
        std = np.std(batch, axis=0)
        data_normalized[i:i + batch_size] = (batch - mean) / std
    return data_normalized """
    
    
    # Normalización de los datos de espiking por lotes
def normalize_in_batches(data, batch_size=1000):
    data_normalized = np.empty_like(data)
    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        mean = np.mean(batch, axis=0)
        std = np.std(batch, axis=0)
        data_normalized[i:i + batch_size] = (batch - mean) / std
        logging.info(f"Lote {i // batch_size + 1}/{num_batches} normalizado")
    return data_normalized
    

spiking_data_array = normalize_in_batches(spiking_data_array)

spiking_data_tensor = torch.tensor(spiking_data_array)

spiking_data_tensor_flat = spiking_data_tensor.view(-1, 3, 64, 64)
labels_flat = np.repeat(labels, 5)

X_train, X_temp, y_train, y_temp = train_test_split(spiking_data_tensor_flat, labels_flat, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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

model_csnn = DeepCSNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model_csnn.parameters(), lr=0.0001)

num_epochs = 1
batch_size = 16  # Reducido para evitar el problema de memoria
n_samples = X_train.shape[0]
n_batches = n_samples // batch_size


with tracker:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model_csnn.train()
        for i in tqdm(range(n_batches)):
            optimizer.zero_grad()
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            outputs = model_csnn(batch_X)
            loss = criterion(outputs.squeeze(), torch.tensor(batch_y, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logging.info(f"Epoch {epoch + 1}, Loss: {epoch_loss / n_batches}")
        print_memory_usage()  # Monitorear el uso de memoria

model_csnn.eval()

def predict_in_batches(model, data, batch_size):
    predictions = []
    n_samples = data.shape[0]
    n_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_data = data[start_idx:end_idx]
            batch_preds = model(batch_data).squeeze().cpu().numpy()
            predictions.extend(batch_preds)
    return np.array(predictions)

# Predecir en lotes
y_pred_val = predict_in_batches(model_csnn, X_val, batch_size=16)

mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

logging.info("Métricas en el conjunto de validación (CSNN):")
logging.info(f"MSE: {mse_val}")
logging.info(f"MAE: {mae_val}")
logging.info(f"R^2: {r2_val}")
