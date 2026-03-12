"""Simple Identity Test"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple 2-ball crossing dataset
print("Creating crossing dataset...")

# 50% crossing, 50% non-crossing
data = []
for i in range(5000):
    # Ball positions
    if i < 2500:  # Crossing
        b1_x, b1_y = 5.0 + i * 0.01, 16.0
        b2_x, b2_y = 27.0 - i * 0.01, 16.0
    else:  # Non-crossing
        b1_x, b1_y = 10.0, 10.0 + i * 0.001
        b2_x, b2_y = 22.0, 22.0 + i * 0.001
    
    # Image: two white dots on black
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[int(b1_y), int(b1_x)] = [1, 1, 1]
    img[int(b2_y), int(b2_x)] = [1, 1, 1]
    
    # Target: ball1 position at t+3
    if i < 2500:
        target = [(5.0 + (i+3) * 0.01) / 32, 16.0 / 32]
    else:
        target = [10.0 / 32, (10.0 + (i+3) * 0.001) / 32]
    
    data.append((img, target, 1 if i < 2500 else 0))

print(f"Dataset: {len(data)} samples")

# Extract arrays
images = np.array([d[0] for d in data])
targets = np.array([d[1] for d in data])
crossing = np.array([d[2] for d in data])

# To tensors
X = torch.FloatTensor(images).permute(0, 3, 1, 2)
Y = torch.FloatTensor(targets)

# Simple model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(32*8*8, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        return self.fc(h), h

print("Training...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(10):
    for i in range(0, len(X), 32):
        p, _ = m(X[i:i+32])
        loss = F.mse_loss(p, Y[i:i+32])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  Epoch {ep+1}")

# Test
m.eval()
with torch.no_grad():
    p, _ = m(X)
    mse_all = F.mse_loss(p, Y).item()
    mse_c = F.mse_loss(p[crossing==1], Y[crossing==1]).item()
    mse_n = F.mse_loss(p[crossing==0], Y[crossing==0]).item()

print(f"\nResults:")
print(f"  MSE (all): {mse_all:.4f}")
print(f"  MSE (crossing): {mse_c:.4f}")
print(f"  MSE (non-crossing): {mse_n:.4f}")
