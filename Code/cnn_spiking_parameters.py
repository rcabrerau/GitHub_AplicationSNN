import torch
import torch.nn as nn

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

# Contar los parámetros
total_params = sum(p.numel() for p in model_csnn.parameters())
print(f'Total parameters: {total_params}')   ##Total parameters: 4615809
