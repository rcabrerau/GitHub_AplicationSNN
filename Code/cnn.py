import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from codecarbon import EmissionsTracker


# Inicializar el rastreador de emisiones
tracker = EmissionsTracker()

# Definir las transformaciones para los datos
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Ruta al directorio con las imágenes
data_dir = "E:\MASTER UOC\AULAS_4TO_SEMESTRE\TFM\AplicationSNN\datasetConvertion_CNN_to_SNN\IMG"

# Cargar el conjunto de datos
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Separar el conjunto de datos en entrenamiento, prueba y validación
train_size = int(0.8 * len(dataset))
temp_size = len(dataset) - train_size
val_size = int(0.5 * temp_size)
test_size = temp_size - val_size

train_dataset, temp_dataset = random_split(dataset, [train_size, temp_size])
val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

# Crear DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definir el modelo de red neuronal
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Inicializar el modelo
model = CNN()

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
num_epochs = 10
# Iniciar el seguimiento de emisiones para el entrenamiento
with tracker:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for batch_X, batch_y in tqdm(train_loader):
            optimizer.zero_grad()  # Reiniciar los gradientes

        # Pasar los datos al modelo
        outputs = model(batch_X)
        # Calcular la pérdida y realizar la retropropagación
        loss = criterion(outputs.squeeze(), batch_y.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

# Evaluar el modelo en el conjunto de validación
model.eval()
y_pred_val = []
y_true_val = []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        outputs = model(batch_X).squeeze()
        y_pred_val.extend(outputs.tolist())
        y_true_val.extend(batch_y.tolist())

# Calcular métricas en el conjunto de validación
mse_val = mean_squared_error(y_true_val, y_pred_val)
mae_val = mean_absolute_error(y_true_val, y_pred_val)
r2_val = r2_score(y_true_val, y_pred_val)

print("Métricas en el conjunto de validación:")
print(f"MSE: {mse_val}")
print(f"MAE: {mae_val}")
print(f"R^2: {r2_val}")