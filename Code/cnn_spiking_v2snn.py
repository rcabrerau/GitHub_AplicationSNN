

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import h5py
import numpy as np
import snntorch as snn

# Definir el mapeo de etiquetas
label_map = {'center': 0, 'left': 1, 'right': 2}

# Clase para cargar los datos desde el archivo HDF5
class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.spiking_data_list = []
        self.labels = []

        with h5py.File(file_path, 'r') as f:
            for group_name in f:
                if 'spiking_data' in f[group_name]:
                    spiking_data = f[group_name]['spiking_data'][:]
                    self.spiking_data_list.append(spiking_data)
                    label = label_map[f[group_name].attrs['label']]
                    self.labels.append(label)

        self.spiking_data_array = np.array(self.spiking_data_list, dtype=np.float32)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spiking_data_array[idx], self.labels[idx]

# Definir la arquitectura del modelo
class SpikingNet(nn.Module):
    def __init__(self, beta=0.95):
        super(SpikingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4)
        print("Dimensiones despues de self.conv1:", self.conv1)
        self.leaky1 = snn.Leaky(beta=beta, init_hidden=True)
        print("Dimensiones despues de self.leaky1:", self.leaky1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        print("Dimensiones despues de self.conv2:", self.conv2)        
        self.leaky2 = snn.Leaky(beta=beta, init_hidden=True)
        print("Dimensiones despues de self.leaky2:", self.leaky2)   
        self.pool = nn.MaxPool2d(2)
        print("Dimensiones despues de pool:", self.pool)
        self.flatten = nn.Flatten()
        print("Dimensiones despues de flatten:", self.flatten)
        self.fc1 = nn.Linear(64 * 14 * 14, 10)
        print("Dimensiones despues de self.fc1:", self.fc1)
        self.leaky3 = snn.Leaky(beta=beta, init_hidden=True, output=True)
        self.fc2 = nn.Linear(10, 3)  # Output layer with 3 classes

    def forward(self, x):
        x = self.pool(self.leaky1(self.conv1(x)))
        x = self.pool(self.leaky2(self.conv2(x)))
        x = self.flatten(x)
        print("Dimensiones antes de fc1:", x.shape)
        x = self.fc1(x)
        x, _ = self.leaky3(x)
        x = x.mean(dim=0)  # Average across the temporal dimension
        #x = self.leaky3(self.fc1(x))
        print("Dimensiones después de fc1:", x.shape)
        x = self.fc2(x)
        return x

# Función para calcular las métricas
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

# Ruta al archivo HDF5
file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_22062024_1529_test.h5"

# Crear instancia del dataset
dataset = CustomDataset(file_path)

# Dividir el dataset en conjuntos de entrenamiento, validación y prueba
train_set, temp_set = train_test_split(dataset, test_size=0.2, random_state=42)
val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

# Crear dataloaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Crear instancia del modelo
model = SpikingNet()

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        # Reshape de los datos para que coincidan con la forma esperada por el modelo
        inputs = inputs.view(-1, 3, 64, 64)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
    
    
# Evaluación del modelo en el conjunto de validación
model.eval()
y_true_val = []
y_pred_val = []
for inputs, labels in val_loader:
    # Reshape de los datos para que coincidan con la forma esperada por el modelo
    inputs = inputs.view(-1, 3, 64, 64)
    outputs = model(inputs)
    y_true_val.extend(labels.cpu().numpy())
    y_pred_val.extend(outputs.argmax(dim=1).cpu().numpy())

# Calcular métricas en el conjunto de validación
mse_val, mae_val, r2_val = compute_metrics(y_true_val, y_pred_val)

# Imprimir las métricas en el conjunto de validación
print("MSE en conjunto de validación:", mse_val)
print("MAE en conjunto de validación:", mae_val)
print("R^2 en conjunto de validación:", r2_val)

