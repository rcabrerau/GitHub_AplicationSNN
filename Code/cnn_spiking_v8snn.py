import h5py
import numpy as np
import torch
import logging
import sys
import snntorch as snn
from snntorch import spikegen
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Habilitar la detección de anomalías
torch.autograd.set_detect_anomaly(True)

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_snn_import_data.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger()

file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_22062024_1529_test.h5"

label_map = {'center': 0, 'left': 1, 'right': 2}

def load_data(file_path):
    spiking_data_list = []
    labels = []

    with h5py.File(file_path, 'r') as f:
        for group_name in f:
            if 'spiking_data' in f[group_name]:
                spiking_data = f[group_name]['spiking_data'][:]
                spiking_data_list.append(spiking_data)
                label = label_map[f[group_name].attrs['label']]
                labels.append(label)

    spiking_data_array = np.array(spiking_data_list, dtype=np.float32)
    spiking_data_tensor = torch.tensor(spiking_data_array)
    spiking_data_tensor_flat = spiking_data_tensor.view(-1, 3, 64, 64)
    labels_flat = np.repeat(labels, 5)

    # Dividir los datos en entrenamiento, prueba y validación
    X_train, X_temp, y_train, y_temp = train_test_split(spiking_data_tensor_flat, labels_flat, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logger.info(f"Shape of spiking_data_tensor_flat: {spiking_data_tensor_flat.shape}")
    logger.info(f"Shape of labels_flat: {labels_flat.shape}")

    return X_train, y_train, X_test, y_test, X_val, y_val

# Cargar los datos
X_train, y_train, X_test, y_test, X_val, y_val = load_data(file_path)

# Verificar las formas de los datos cargados
logger.info(f"Shape of X_train: {X_train.shape}")
logger.info(f"Shape of y_train: {y_train.shape}")
logger.info(f"Shape of X_test: {X_test.shape}")
logger.info(f"Shape of y_test: {y_test.shape}")
logger.info(f"Shape of X_val: {X_val.shape}")
logger.info(f"Shape of y_val: {y_val.shape}")

# Definir el número de pasos de tiempo
num_steps = 5

# Generar picos
spike_train = spikegen.latency(X_train, num_steps=num_steps, normalize=True)
spike_test = spikegen.latency(X_test, num_steps=num_steps, normalize=True)
spike_val = spikegen.latency(X_val, num_steps=num_steps, normalize=True)

# Verificar las formas de los datos de picos
logger.info(f"Shape of spike_train: {spike_train.shape}")
logger.info(f"Shape of spike_test: {spike_test.shape}")
logger.info(f"Shape of spike_val: {spike_val.shape}")

# Aplanar las dimensiones temporales
spike_train_flat = spike_train.view(-1, 3, 64, 64)
spike_test_flat = spike_test.view(-1, 3, 64, 64)
spike_val_flat = spike_val.view(-1, 3, 64, 64)

# Replicar etiquetas para coincidir con el número de pasos de tiempo
y_train_repeated = torch.tensor(np.repeat(y_train, num_steps), dtype=torch.long)
y_test_repeated = torch.tensor(np.repeat(y_test, num_steps), dtype=torch.long)
y_val_repeated = torch.tensor(np.repeat(y_val, num_steps), dtype=torch.long)

# Verificar las formas de los tensores aplanados y replicados
logger.info(f"Shape of spike_train_flat: {spike_train_flat.shape}")
logger.info(f"Shape of y_train_repeated: {y_train_repeated.shape}")
logger.info(f"Shape of spike_test_flat: {spike_test_flat.shape}")
logger.info(f"Shape of y_test_repeated: {y_test_repeated.shape}")
logger.info(f"Shape of spike_val_flat: {spike_val_flat.shape}")
logger.info(f"Shape of y_val_repeated: {y_val_repeated.shape}")

class SpikingCNN(nn.Module):
    def __init__(self):
        super(SpikingCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*16*16, 128)
        self.fc2 = nn.Linear(128, 3)
        self.lif1 = snn.Leaky(beta=0.95)
        self.lif2 = snn.Leaky(beta=0.95)

    def forward(self, x):
        x = self.conv1(x)
        spk1, mem1 = self.lif1(self.pool(x.clone()))  # Evitar operaciones in-place
        logger.info(f"spk1 shape: {spk1.shape}")
        x = self.conv2(spk1.clone())
        spk2, mem2 = self.lif2(self.pool(x.clone()))  # Evitar operaciones in-place
        logger.info(f"spk2 shape: {spk2.shape}")
        spk2 = spk2.view(spk2.size(0), -1)
        logger.info(f"spk2 reshaped: {spk2.shape}")
        x = self.fc1(spk2)
        x = self.fc2(x)
        return x

model = SpikingCNN()
logger.info("Model created successfully")

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Crear DataLoaders
train_dataset = TensorDataset(spike_train_flat, y_train_repeated)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(spike_val_flat, y_val_repeated)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Verificar las formas de los tensores
logger.info(f"Shape of spike_train_flat: {spike_train_flat.shape}")
logger.info(f"Shape of y_train_repeated: {y_train_repeated.shape}")
logger.info(f"Shape of spike_val_flat: {spike_val_flat.shape}")
logger.info(f"Shape of y_val_repeated: {y_val_repeated.shape}")

num_epochs = 10  # Cambia a más épocas según sea necesario

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        logger.info(f"Batch {batch_idx+1}/{len(train_loader)}")
        logger.info(f"Images shape: {images.shape}")
        logger.info(f"Labels shape: {labels.shape}")

        optimizer.zero_grad()
        outputs = model(images.clone())  # Evitar operaciones in-place
        logger.info(f"Outputs shape: {outputs.shape}")
        
        labels = labels.long()
        
        loss = criterion(outputs, labels)
        logger.info(f"Loss: {loss.item()}")
        
        loss.backward()  # No necesitamos retain_graph aquí
        logger.info("Backward pass done")
                
        optimizer.step()
        logger.info("Optimizer step done")

        epoch_loss += loss.item()
        
    avg_epoch_loss = epoch_loss / len(train_loader)
    logger.info(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}")

    # Validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.clone())
            loss = criterion(outputs, labels.long())
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")
    scheduler.step(avg_val_loss)

# Guardar el modelo
torch.save(model.state_dict(), "spiking_cnn_model.pth")

# Evaluar el modelo
test_dataset = TensorDataset(spike_test_flat, y_test_repeated)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluación del modelo
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.clone())
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

# Calcular métricas de evaluación
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

logger.info(f"Test MSE: {mse:.4f}")
logger.info(f"Test MAE: {mae:.4f}")
logger.info(f"Test R^2: {r2:.4f}")
