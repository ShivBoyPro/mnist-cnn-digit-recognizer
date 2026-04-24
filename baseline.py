import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# Set Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data Loading and Preprocessing
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Reshape to (Batch, Channel, Height, Width) and Normalize to [0, 1]
X_raw = torch.tensor(train_df.iloc[:, 1:].values).float().view(-1, 1, 28, 28) / 255.0
y_raw = torch.tensor(train_df.iloc[:, 0].values).long()
X_test = torch.tensor(test_df.values).float().view(-1, 1, 28, 28) / 255.0

# 2. Define Architecture (EliteCNN)
class EliteCNN(nn.Module):
    def __init__(self):
        super(EliteCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 3. Training Setup
model = EliteCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
loader = DataLoader(TensorDataset(X_raw, y_raw), batch_size=64, shuffle=True)

# 4. Training Loop
print("Starting Training (15 Epochs)...")
model.train()
for epoch in range(15):
    running_loss = 0.0
    for i, (batch_X, batch_y) in enumerate(loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Simple Data Augmentation: Random slight rotations on the fly
        # (This helps the model generalize to messy handwriting)
        if np.random.rand() > 0.5:
            batch_X = transforms.functional.rotate(batch_X, angle=np.random.uniform(-10, 10))

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch [{epoch+1}/15], Step [{i+1}/{len(loader)}], Loss: {running_loss/100:.4f}")
            running_loss = 0.0

# 5. Generate Submission
print("Training complete. Generating submission.csv...")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.to(device))
    predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()

submission = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'Label': predictions})
submission.to_csv('submission.csv', index=False)
print("File saved! Ready for Kaggle upload.")