import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from codecarbon import EmissionsTracker
import snntorch as snn
from snntorch import spikegen
import logging
import sys

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_csnn.txt"),
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

    X_train, X_temp, y_train, y_temp = train_test_split(spiking_data_tensor_flat, labels_flat, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_test, y_test, X_val, y_val

# Cargar los datos
X_train, y_train, X_test, y_test, X_val, y_val = load_data(file_path)
num_steps = 5

# Generar picos
spike_train = spikegen.latency(X_train, num_steps=num_steps, normalize=True)
spike_test = spikegen.latency(X_test, num_steps=num_steps, normalize=True)

# Aplanar las dimensiones temporales
spike_train_flat = spike_train.view(-1, 3, 64, 64)
spike_test_flat = spike_test.view(-1, 3, 64, 64)

# Replicar etiquetas para coincidir con el número de pasos de tiempo
y_train_repeated = y_train.repeat(num_steps)
y_test_repeated = y_test.repeat(num_steps)

# Convertir etiquetas a tensores de PyTorch
y_train_repeated = torch.tensor(y_train_repeated, dtype=torch.long)
y_test_repeated = torch.tensor(y_test_repeated, dtype=torch.long)

# Verificar las formas de los tensores
logger.info(f"Shape of spike_train_flat: {spike_train_flat.shape}")
logger.info(f"Shape of y_train_repeated: {y_train_repeated.shape}")
logger.info.f("Shape of spike_test_flat: {spike_test_flat.shape}")
logger.info.f("Shape of y_test_repeated: {y_test_repeated.shape}")

# Asegurarnos de que las etiquetas coincidan correctamente
if spike_train_flat.shape[0] != y_train_repeated.shape[0]:
    logger.error(f"Mismatch in training data shapes: spike_train_flat.shape[0]={spike_train_flat.shape[0]}, y_train_repeated.shape[0]={y_train_repeated.shape[0]}")
if spike_test_flat.shape[0] != y_test_repeated.shape[0]:
    logger.error(f"Mismatch in testing data shapes: spike_test_flat.shape[0]={spike_test_flat.shape[0]}, y_test_repeated.shape[0]={y_test_repeated.shape[0]}")

# Crear DataLoaders
try:
    train_dataset = TensorDataset(spike_train_flat, y_train_repeated)
    test_dataset = TensorDataset(spike_test_flat, y_test_repeated)
except AssertionError as e:
    logger.error(f"Assertion error: {e}")
    logger.error.f("Shape of spike_train_flat: {spike_train_flat.shape}")
    logger.error.f("Shape of y_train_repeated: {y_train_repeated.shape}")
    raise

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

logger.info(f"Data preprocessed and split: spike_train_flat={spike_train_flat.shape}, spike_test_flat={spike_test_flat.shape}")

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
        spk1, mem1 = self.lif1(self.pool(self.conv1(x.clone())))  # Evitar operaciones in-place
        logger.info(f"spk1 shape: {spk1.shape}")
        spk2, mem2 = self.lif2(self.pool(self.conv2(spk1.clone())))  # Evitar operaciones in-place
        logger.info(f"spk2 shape: {spk2.shape}")
        spk2 = spk2.view(spk2.size(0), -1)
        logger.info(f"spk2 reshaped: {spk2.shape}")
        x = self.fc1(spk2)
        x = self.fc2(x)
        return x

model = SpikingCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

logger.info("Model created successfully")

# Habilitar la detección de anomalías
torch.autograd.set_detect_anomaly(True)

# Entrenamiento del modelo
#tracker = EmissionsTracker()
#tracker.start()

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        logger.info(f"Batch {batch_idx+1}/{len(train_loader)}")
        logger.info(f"Images shape: {images.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        
        optimizer.zero_grad()
        
        outputs = model(images)
        logger.info(f"Outputs shape: {outputs.shape}")
        
        loss = criterion(outputs, labels)
        logger.info.f("Loss: {loss.item()}")

        # Habilitar detección de anomalías solo para esta parte
        with torch.autograd.detect_anomaly():
            loss.backward()
        
        logger.info("Backward pass done")
        
        optimizer.step()
        logger.info("Optimizer step done")
        
        # Limpieza del gráfico después de cada iteración
        del loss, outputs
        torch.cuda.empty_cache()
        logger.info("Cache cleaned")

#emissions = tracker.stop()
#logger.info(f"Training completed with emissions: {emissions:.4f} kg CO2")

# Guardar el modelo
torch.save(model.state_dict(), 'spiking_cnn.pth')

# Evaluación del modelo
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

logger.info(f"Test MSE: {mse:.4f}")
logger.info(f"Test MAE: {mae:.4f}")
logger.info.f("Test R^2: {r2:.4f}")
