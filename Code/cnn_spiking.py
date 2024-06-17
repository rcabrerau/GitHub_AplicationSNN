import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import logging
import sys
import psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from codecarbon import EmissionsTracker

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_csnn.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])

tracker = EmissionsTracker()
file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_16062024_0029.h5"

spiking_data_list = []
labels = []

label_map = {'center': 0, 'left': 1, 'right': 2}  # Define un mapeo de etiquetas a números

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(4, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_skip = nn.GroupNorm(4, out_channels)
        
    def forward(self, x):
        identity = self.bn_skip(self.skip(x))
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return torch.relu(out)

class ImprovedDeepCSNN(nn.Module):
    def __init__(self):
        super(ImprovedDeepCSNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.GroupNorm(4, 32)
        self.res1 = ResidualBlock(32, 64)
        self.res2 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool(self.res1(x))
        x = self.pool(self.res2(x))
        x = torch.flatten(x, 1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model_csnn = ImprovedDeepCSNN()

criterion = nn.MSELoss()
optimizer = optim.AdamW(model_csnn.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

num_epochs = 10
batch_size = 16
n_samples = X_train.shape[0]
n_batches = n_samples // batch_size

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

y_pred_val = predict_in_batches(model_csnn, X_val, batch_size)

with tracker:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model_csnn.train()        
        for i in tqdm(range(n_batches), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):           
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
        avg_epoch_loss = epoch_loss / n_batches
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}")
        print_memory_usage()
        
        model_csnn.eval()
        y_pred_val = predict_in_batches(model_csnn, X_val, batch_size)
        val_loss = mean_squared_error(y_val, y_pred_val)  # Usar etiquetas codificadas
        scheduler.step(val_loss)
        logging.info(f"Learning rate: {scheduler.get_last_lr()}")  # Registrar el ritmo de aprendizaje actual

model_csnn.eval()

mse_val = mean_squared_error(y_val, y_pred_val)  # Usar etiquetas codificadas
mae_val = mean_absolute_error(y_val, y_pred_val)  # Usar etiquetas codificadas
r2_val = r2_score(y_val, y_pred_val)  # Usar etiquetas codificadas

logging.info("Validation Metrics (CSNN):")
logging.info(f"MSE: {mse_val}")
logging.info(f"MAE: {mae_val}")
logging.info(f"R^2: {r2_val}")