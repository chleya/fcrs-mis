"""
Identity Test - FINAL v4
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("IDENTITY TEST - FINAL v4")
print("="*60)

def generate_data(n=2000):
    all_t0, all_target = [], []
    
    for _ in range(n):
        x_a, y_a, vx_a, vy_a = 10.0, 16.0, 1.5, 0.0
        x_b, y_b, vx_b, vy_b = 22.0, 16.0, -1.5, 0.0
        
        # t0: red/blue balls
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[int(y_a), int(x_a)] = [1, 0, 0]
        img[int(y_b), int(x_b)] = [0, 0, 1]
        all_t0.append(img)
        
        # Simulate to t10
        for _ in range(10):
            x_a, y_a = x_a + vx_a * 0.8, y_a + vy_a * 0.8
            if x_a < 3 or x_a > 29: vx_a *= -1
            if y_a < 3 or y_a > 29: vy_a *= -1
            x_b = ((x_b - 3) % 26) + 3
            y_b = ((y_b - 3) % 26) + 3
        
        all_target.append([x_a / 32, y_a / 32])
    
    return np.array(all_t0), np.array(all_target)

print("\n1. Generating data...")
X, Y = generate_data(2000)
X = torch.FloatTensor(X).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slot = nn.Parameter(torch.randn(2, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        
    def forward(self, x):
        h = self.enc(x)  # (B, 64, 8, 8)
        h = h.mean(dim=[2, 3])  # (B, 64)
        h = h.unsqueeze(1) + self.slot.unsqueeze(0)  # (B, 2, 64)
        return self.predict(h[:, 0])

print("\n2. Training...")

results = {}

# Baseline
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p = m(X[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
m.eval()
with torch.no_grad():
    results['Baseline'] = F.mse_loss(m(X), Y).item()

# Slot
m = SlotModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p = m(X[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
m.eval()
with torch.no_grad():
    results['Slot'] = F.mse_loss(m(X), Y).item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
for name, mse in results.items():
    print(f"{name}: MSE = {mse:.4f}")
