"""
Identity Test - RANDOM velocities (harder)
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
print("IDENTITY TEST - RANDOM VELOCITIES")
print("="*60)

def generate_data(n=3000):
    all_t0, all_target = [], []
    
    for _ in range(n):
        # Random initial velocities
        vx_a = random.uniform(-2, 2)
        vy_a = random.uniform(-2, 2)
        vx_b = random.uniform(-2, 2)
        vy_b = random.uniform(-2, 2)
        
        # Ensure they move toward each other
        x_a, y_a = 10.0, random.uniform(10, 22)
        x_b, y_b = 22.0, random.uniform(10, 22)
        
        # t0: colored
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[int(y_a), int(x_a)] = [1, 0, 0]
        img[int(y_b), int(x_b)] = [0, 0, 1]
        all_t0.append(img)
        
        # Simulate to t10 with DIFFERENT rules
        for _ in range(10):
            x_a, y_a = x_a + vx_a * 0.8, y_a + vy_a * 0.8
            if x_a < 3 or x_a > 29: vx_a *= -1
            if y_a < 3 or y_a > 29: vy_a *= -1
            
            # Ball B wraps
            x_b = ((x_b - 3) % 26) + 3
            y_b = ((y_b - 3) % 26) + 3
        
        all_target.append([x_a / 32, y_a / 32])
    
    return np.array(all_t0), np.array(all_target)

print("\n1. Generating data...")
X, Y = generate_data(3000)
X = torch.FloatTensor(X).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)
print(f"   X: {X.shape}, Y: {Y.shape}")

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
        h = self.enc(x).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slot.unsqueeze(0)
        return self.predict(h[:, 0])

print("\n2. Training...")

results = {}

# Baseline
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(12):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p = m(X[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
results['Baseline'] = F.mse_loss(m(X), Y).item()

# Slot
m = SlotModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(12):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p = m(X[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
results['Slot'] = F.mse_loss(m(X), Y).item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
for name, mse in results.items():
    print(f"{name}: MSE = {mse:.4f}")
