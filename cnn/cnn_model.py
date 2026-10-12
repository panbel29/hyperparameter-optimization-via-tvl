import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# MNIST — architecture
# ─────────────────────────────────────────────────────────────────────────────

class CNN_MNIST(nn.Module):
    _FLAT = 20 * 12 * 12   # number of units = 2880

    def __init__(self, dropout_prob=None, n_classes=10):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0) # 1st convolutional layer: 20 filters, 5x5 kernel, no padding
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max pooling layer: 2x2 window, stride 2 
        self.fc1 = nn.Linear(self._FLAT, 512) # Fully connected layer mapping from flattened output to 512 units
        self.dropout = (nn.Dropout(p=dropout_prob) # Dropout layer (active if dropout_prob is not None, otherwise Identity)
                        if dropout_prob is not None else nn.Identity())
        self.fc2 = nn.Linear(512, n_classes) # Fully connected layer mapping from 512 units to n_classes (10 for MNIST)

    def forward(self, x):
        x = self.conv1(x) # (B, 1, 28, 28) -> (B, 20, 24, 24)
        x = F.relu(x)
        x = self.pool(x) # (B, 20, 24, 24) -> (B, 20, 12, 12)

        x = torch.flatten(x, 1) # (B, 20, 12, 12) -> (B, 2880)

        x = self.fc1(x) # (B, 2880) -> (B, 512)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) # (B, 512) -> (B, 10)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# FMNIST — architecture
# ─────────────────────────────────────────────────────────────────────────────

class CNN_FMNIST(nn.Module):
    _FLAT = 64 * 7 * 7   # number of units = 3136

    def __init__(self, dropout_prob=None, n_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1,  16, kernel_size=3, stride=1, padding=1) # 1st convolution layer: 16 filters, 3x3 same-padding        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 2nd convolution layer: 32 filters, 3x3 same-padding       
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 3rd convolution layer: 64 filters, 3x3 same-padding
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2) # Pooling layer 
        self.fc1     = nn.Linear(self._FLAT, 256) # Fully connected layer
        self.dropout = (nn.Dropout(p=dropout_prob) # Dropout layer (active if dropout_prob is not None, otherwise Identity)
                        if dropout_prob is not None else nn.Identity())
        self.fc2     = nn.Linear(256, n_classes) # Fully connected layer mapping to n_classes (10 for FMNIST)

    def forward(self, x):
        x = self.conv1(x) # (B, 1, 28, 28) -> (B, 16, 28, 28)
        x = F.relu(x)
        x = self.pool(x) # (B, 16, 28, 28) -> (B, 16, 14, 14)

        x = self.conv2(x) # (B, 16, 14, 14) -> (B, 32, 14, 14)
        x = F.relu(x)
        x = self.pool(x) # (B, 32, 14, 14) -> (B, 32, 7, 7)

        x = self.conv3(x) # (B, 32, 7, 7) -> (B, 64, 7, 7)
        x = F.relu(x)

        x = torch.flatten(x, 1) # (B, 64, 7, 7) -> (B, 3136)

        x = self.fc1(x) # (B, 3136) -> (B, 256)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) # (B, 256) -> (B, 10)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 — architecture
# ─────────────────────────────────────────────────────────────────────────────

class CNN_CIFAR10(nn.Module):
    _FLAT = 64 * 4 * 4   # number of units = 1024 

    def __init__(self, dropout_prob=None, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  16, kernel_size=3, stride=1, padding=1) # 1st convolutional layer: 16 filters, 3x3 same-padding 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 2nd convolutional layer: 32 filters, 3x3 same-padding   
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 3rd convolutional layer: 64 filters, 3x3 same-padding
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # 4th convolutional layer: 64 filters   
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2) # Max pooling layer: 2x2 window, stride 2         
        self.fc1     = nn.Linear(self._FLAT, 256) # Fully connected layer
        self.dropout = (nn.Dropout(p=dropout_prob) # Dropout layer (active if dropout_prob is not None, otherwise Identity)
                        if dropout_prob is not None else nn.Identity())
        self.fc2     = nn.Linear(256, n_classes) # Fully connected layer mapping to n_classes (10 for CIFAR-10)

    def forward(self, x):
        # Block 1: conv → ReLU → pool   32x32 → 16x16
        x = self.conv1(x) # (B, 3, 32, 32) -> (B, 16, 32, 32)
        x = F.relu(x)
        x = self.pool(x) # (B, 16, 32, 32) -> (B, 16, 16, 16)

        x = self.conv2(x) # (B, 16, 16, 16) -> (B, 32, 16, 16)
        x = F.relu(x)
        x = self.pool(x) # (B, 32, 16, 16) -> (B, 32, 8, 8)

        x = self.conv3(x) # (B, 32, 8, 8) -> (B, 64, 8, 8)
        x = F.relu(x)
        x = self.pool(x) # (B, 64, 8, 8) -> (B, 64, 4, 4)

        x = self.conv4(x) # (B, 64, 4, 4) -> (B, 64, 4, 4)
        x = F.relu(x)
        x = torch.flatten(x, 1) # (B, 64, 4, 4) -> (B, 1024)

        x = self.fc1(x) # (B, 1024) -> (B, 256)
        x = F.relu(x) 

        x = self.dropout(x)

        x = self.fc2(x) # (B, 256) -> (B, 10)

        return x

# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn(dataset: str, dropout_prob=None, n_classes: int = 10) -> nn.Module:
    """
    Return the correct CNN for *dataset*.
    """
    d = dataset.upper()
    if d == 'MNIST':
        return CNN_MNIST(dropout_prob=dropout_prob, n_classes=n_classes)
    elif d == 'FMNIST':
        return CNN_FMNIST(dropout_prob=dropout_prob, n_classes=n_classes)
    elif d == 'CIFAR10':
        return CNN_CIFAR10(dropout_prob=dropout_prob, n_classes=n_classes)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose MNIST, FMNIST, or CIFAR10.")
