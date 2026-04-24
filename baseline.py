import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Advanced Data Prep with Augmentation
train = pd.read_csv('train.csv')
X_raw = torch.tensor(train.iloc[:, 1:].values).float().view(-1, 1, 28, 28) / 255.0
y_raw = torch.tensor(train.iloc[:, 0].values).long()

# Augmentation makes the model "smarter" about messy handwriting
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1))
])

# 2. Higher-Capacity CNN
class EliteCNN(nn.Module):
    def __init__(self):
        super(EliteCNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.fc_stack = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        return self.fc_stack(x)

model = EliteCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Learning rate scheduler: slows down as it gets closer to the goal
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)