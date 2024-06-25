import h5py
import numpy as np
import torch
import logging
import sys
import snntorch as snn
import snntorch.functional as SF
import snntorch.utils as utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import time
import statistics
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()

# Habilitar la detección de anomalías
torch.autograd.set_detect_anomaly(True)

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_snn.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger()

# Forzar el uso de CPU
device = torch.device("cpu")

#file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_22062024_1529_test.h5"
file_path = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN_to_SNN/SPIKING_labels/spiking_data_labels_16062024_0029.h5"

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

    return X_train, y_train, X_test, y_test, X_val, y_val

# Cargar los datos (usando todos los datos)
X_train, y_train, X_test, y_test, X_val, y_val = load_data(file_path)

# Aplanar las dimensiones temporales y reestructurar adecuadamente
spike_train_flat = X_train.view(-1, 3, 64, 64)  # Número total de muestras = num_steps * número de ejemplos
spike_test_flat = X_test.view(-1, 3, 64, 64)
spike_val_flat = X_val.view(-1, 3, 64, 64)

# Replicar etiquetas para coincidir con el número de pasos de tiempo
y_train_repeated = torch.tensor(np.repeat(y_train, 1), dtype=torch.long)
y_test_repeated = torch.tensor(np.repeat(y_test, 1), dtype=torch.long)
y_val_repeated = torch.tensor(np.repeat(y_val, 1), dtype=torch.long)

# neuron and simulation parameters
beta = 0.95

# Calcula las dimensiones después de la última capa de agrupación (pooling)
def calculate_conv_output_size(input_size, kernel_size, padding=0, stride=1):
    return (input_size - kernel_size + 2 * padding) // stride + 1

# Dimensiones de entrada
input_height = input_width = 64  # Asumiendo imágenes de 64x64

# Primera capa convolucional y pooling
out_height = calculate_conv_output_size(input_height, 4)  # kernel_size=4
out_width = calculate_conv_output_size(input_width, 4)
out_height = calculate_conv_output_size(out_height, 2, stride=2)  # pooling
out_width = calculate_conv_output_size(out_width, 2, stride=2)

# Segunda capa convolucional y pooling
out_height = calculate_conv_output_size(out_height, 3)  # kernel_size=3
out_width = calculate_conv_output_size(out_width, 3)
out_height = calculate_conv_output_size(out_height, 2, stride=2)  # pooling
out_width = calculate_conv_output_size(out_width, 2, stride=2)

# Número de características de entrada para la primera capa lineal
linear_input_features = 64 * out_height * out_width  # 64 es el número de canales de salida de la última convolución

print(f'Output size: {out_height}x{out_width}, Total features: {linear_input_features}')

# Define la red neuronal
scnn_net = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=4),
    snn.Leaky(beta=0.95, init_hidden=True),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, kernel_size=3),
    snn.Leaky(beta=0.95, init_hidden=True),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(linear_input_features, 3),  # Ajustar el número de características aquí
    snn.Leaky(beta=0.95, init_hidden=True, output=True)
).to(device)

# Define el optimizador y la función de pérdida
optimizer = torch.optim.Adam(scnn_net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = nn.CrossEntropyLoss()

# Crear DataLoaders
train_dataset = TensorDataset(spike_train_flat, y_train_repeated)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(spike_val_flat, y_val_repeated)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = TensorDataset(spike_test_flat, y_test_repeated)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def forward_pass_with_mem(net, data):
    spk_rec = []
    mem_rec = []
    utils.reset(net)  # Resetea los estados ocultos para todas las neuronas LIF en net

    for step in range(data.size(0)):  # data.size(0) = número de pasos de tiempo
        data_step = data[step].unsqueeze(0)  # Agregar dimensión de batch
        spk_out, mem_out = net(data_step)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

start_time = time.time()

num_epochs = 10     # Puedes ajustar el número de épocas aquí

loss_hist = []
acc_hist = []

# training loop
total_steps = num_epochs * len(train_loader)
current_step = 0

with tracker:
    with tqdm(total=total_steps, desc="Training Progress", unit="step") as pbar:
        for epoch in range(num_epochs):
            for i, (data, targets) in enumerate(iter(train_loader)):
                data = data.to(device)
                targets = targets.to(device)

                scnn_net.train()
                spk_rec, _ = forward_pass_with_mem(scnn_net, data)
                loss_val = loss_fn(spk_rec.view(-1, 3), targets)

                # Gradient calculation + weight update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Store loss history for future plotting
                loss_hist.append(loss_val.item())

                # Calculate accuracy rate and then append it to accuracy history
                acc = SF.accuracy_rate(spk_rec, targets)
                acc_hist.append(acc)

                # Print loss and accuracy every 4 iterations
                if i % 4 == 0:
                    logger.info(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
                    logger.info(f"Accuracy: {acc * 100:.2f}%\n")

                # Update progress bar
                current_step += 1
                pbar.update(1)

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Convert elapsed time to minutes, seconds, and milliseconds
minutes, seconds = divmod(elapsed_time, 60)
seconds, milliseconds = divmod(seconds, 1)
milliseconds = round(milliseconds * 1000)

# Print the elapsed time
logger.info(f"Elapsed time: {int(minutes)} minutes, {int(seconds)} seconds, {milliseconds} milliseconds")

# Guardar el modelo
torch.save(scnn_net.state_dict(), "spiking_cnn_model.pth")

# Evaluar el modelo en el conjunto de validación
scnn_net.eval()
val_predictions = []
val_targets = []

with torch.no_grad():
    for data, targets in val_loader:
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        spk_rec, mem_rec = forward_pass_with_mem(scnn_net, data)

        # Obtener las clases predichas
        _, predicted_classes = torch.max(mem_rec.view(-1, 3), 1)
        predictions = predicted_classes.cpu().numpy()
        true_values = targets.cpu().numpy()

        val_predictions.extend(predictions)
        val_targets.extend(true_values)

# Convertir listas a arrays de numpy
val_predictions = np.array(val_predictions)
val_targets = np.array(val_targets)

# Calcular las métricas
def calculate_metrics(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    return mse, mae, r2

mse_val, mae_val, r2_val = calculate_metrics(val_targets, val_predictions)

# Registrar las métricas
logger.info("Métricas en el conjunto de validación (CSNN):")
logger.info(f"MSE: {mse_val}")
logger.info(f"MAE: {mae_val}")
logger.info(f"R^2: {r2_val}")

# Evaluar el modelo en el conjunto de prueba
scnn_net.eval()
acc_hist_test = []

# Iterate over batches in the testloader
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        spk_rec, _ = forward_pass_with_mem(scnn_net, data)

        # Calculate accuracy rate
        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist_test.append(acc)

# Print the average accuracy across the testloader
logger.info(f"The average accuracy across the testloader is: {statistics.mean(acc_hist_test)}")
