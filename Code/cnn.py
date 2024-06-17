import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import psutil
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
from codecarbon import EmissionsTracker

# Función para imprimir el uso de memoria
def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

# Configurar el logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log_cnn.txt"),
                        logging.StreamHandler(sys.stdout)
                    ])

# Inicializar el rastreador de emisiones
tracker = EmissionsTracker()

# Ruta al directorio con las imágenes aumentadas
#data_dir = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN/IMG_AUGMENTED"
data_dir = "E:/MASTER UOC/AULAS_4TO_SEMESTRE/TFM/AplicationSNN/datasetConvertion_CNN/IMG_AUGMENTED_TEST"

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Cargar el conjunto de datos
dataset = ImageFolder(root=data_dir, transform=transform)

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

# Definición del modelo
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

class ImprovedDeepCNN(nn.Module):
    def __init__(self):
        super(ImprovedDeepCNN, self).__init__()
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

model_cnn = ImprovedDeepCNN()

criterion = nn.MSELoss()
optimizer = optim.AdamW(model_cnn.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

num_epochs = 2

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.float())
            val_loss += loss.item()
            y_true.extend(batch_y.numpy())
            y_pred.extend(outputs.squeeze().numpy())
    return val_loss / len(val_loader), y_true, y_pred

with tracker:
    for epoch in range(num_epochs):
        avg_epoch_loss = train_epoch(model_cnn, train_loader, criterion, optimizer)
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_epoch_loss}")
        print_memory_usage()

        val_loss, y_val_true, y_val_pred = evaluate_model(model_cnn, val_loader, criterion)
        scheduler.step(val_loss)
        logging.info(f"Validation Loss: {val_loss}")
        logging.info(f"Learning rate: {scheduler.get_last_lr()}")  # Registrar el ritmo de aprendizaje actual


# Evaluar el modelo en el conjunto de validación
model_cnn.eval()
val_loss, y_val_true, y_val_pred = evaluate_model(model_cnn, val_loader, criterion)

mse_val = mean_squared_error(y_val_true, y_val_pred)
mae_val = mean_absolute_error(y_val_true, y_val_pred)
r2_val = r2_score(y_val_true, y_val_pred)

logging.info("Métricas en el conjunto de validación (CNN):")
logging.info(f"MSE: {mse_val}")
logging.info(f"MAE: {mae_val}")
logging.info(f"R^2: {r2_val}")
