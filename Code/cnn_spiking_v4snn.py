#good
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
from torch.utils.data import DataLoader, TensorDataset
from codecarbon import EmissionsTracker
import snntorch as snn
from snntorch import spikegen

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
file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_22062024_1529_test.h5"


spiking_data_list = []
labels = []

label_map = {'center': 0, 'left': 1, 'right': 2}  # Define un mapeo de etiquetas a números

with h5py.File(file_path, 'r') as f:
    for group_name in f:
        if 'spiking_data' in f[group_name]:
            spiking_data = f[group_name]['spiking_data'][:]
            spiking_data_list.append(spiking_data)
            if 'label' in f[group_name].attrs:
                label = label_map[f[group_name].attrs['label']]  # Usa el mapeo para convertir la etiqueta a un número
                labels.append(label)
            else:
                print(f"Label attribute not found in group {group_name}. Skipping this group.")
                logging.warning(f"Label attribute not found in group {group_name}. Skipping this group.")

# Asegúrate de que haya datos y etiquetas
if len(spiking_data_list) == 0 or len(labels) == 0:
    raise ValueError("No se encontraron datos de picos o etiquetas en el archivo HDF5.")

# Convertir los datos y etiquetas a tensores
spiking_data_array = np.array(spiking_data_list, dtype=np.float32)
spiking_data_tensor = torch.tensor(spiking_data_array)
spiking_data_tensor_flat = spiking_data_tensor.view(-1, 3, 64, 64)
labels_flat = np.repeat(labels, 5)

X_train, X_temp, y_train, y_temp = train_test_split(spiking_data_tensor_flat, labels_flat, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_dataset = TensorDataset(X_train, torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(X_val, torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.long))

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SpikingNet(nn.Module):
    def __init__(self, beta=0.95):
        super(SpikingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4)
        self.leaky1 = snn.Leaky(beta=beta, init_hidden=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.leaky2 = snn.Leaky(beta=beta, init_hidden=True)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        
        # Calculate the size of the flattened layer output
        self._to_linear = None
        self.convs(torch.randn(1, 3, 64, 64)) # Pass a dummy input to calculate the output size
        
        self.fc1 = nn.Linear(self._to_linear, 512)  # Adjust this value
        self.fc2 = nn.Linear(512, 3)  # Output layer with 3 classes
        self.dropout = nn.Dropout(0.5)  # Add dropout to avoid overfitting

    def convs(self, x):
        x = self.conv1(x)
        x = self.leaky1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.leaky2(x)
        x = self.pool(x)
        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).shape[1]  # Save the size for Linear layer
        return x
    
    def forward(self, x):
        logging.info(f"Entrada: {x.shape}")
        x = self.convs(x)
        logging.info(f"Después de convs: {x.shape}")
        x = self.flatten(x)
        logging.info(f"Después de flatten: {x.shape}")
        x = self.dropout(torch.relu(self.fc1(x)))
        logging.info(f"Después de fc1: {x.shape}")
        x = self.fc2(x)
        logging.info(f"Después de fc2: {x.shape}")
        return x

model_csnn = SpikingNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model_csnn.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

num_epochs = 1

def predict_in_batches(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data, _ in data_loader:
            batch_preds = model(batch_data).squeeze().cpu().numpy()
            predictions.extend(batch_preds)
    return np.array(predictions)

y_pred_val = predict_in_batches(model_csnn, val_loader)

with tracker:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model_csnn.train()
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            optimizer.zero_grad()
            outputs = model_csnn(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}")
        print_memory_usage()

        y_pred_val = predict_in_batches(model_csnn, val_loader)
        val_loss = mean_squared_error(y_val, y_pred_val)
        scheduler.step(val_loss)
        logging.info(f"Learning rate: {scheduler.get_last_lr()}")

model_csnn.eval()

mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

logging.info("Métricas en el conjunto de validación (CSNN):")
logging.info(f"MSE: {mse_val}")
logging.info(f"MAE: {mae_val}")
logging.info(f"R^2: {r2_val}")