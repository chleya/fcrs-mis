"""
BASELINE vs SLOT - Quick version
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
print("BASELINE vs SLOT QUICK TEST")
print("="*60)

# Generate data
def generate_data(n=2000):
    X0, X5, X10, Y = [], [], [], []
    for _ in range(n):
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        vx_a, vx_b = random.uniform(-2, 2), random.uniform(-2, 2)
        
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]
        img0[int(y), int(x_b)] = [0, 0, 1]
        
        white = [1, 1, 1]
        for _ in range(1, 5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
        
        x_a_t5 = random.uniform(5, 27)
        x_b_t5 = random.uniform(5, 27)
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a_t5)] = white
        img5[int(y), int(x_b_t5)] = white
        
        vx_a, vx_b = random.uniform(-2, 2), random.uniform(-2, 2)
        for _ in range(6, 10):
            x_a_t5 += vx_a * 0.5
            x_b_t5 += vx_b * 0.5
        
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x_a_t5)] = white
        img10[int(y), int(x_b_t5)] = white
        
        X0.append(img0); X5.append(img5); X10.append(img10)
        Y.append(x_a_t5 / 32)
    
    return np.array(X0), np.array(X5), np.array(X10), np.array(Y)

X0, X5, X10, Y = generate_data(2000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"Data: {X0.shape}")

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8*3, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x0, x5, x10):
        return self.fc(torch.cat([self.enc(x0).flatten(1), self.enc(x5).flatten(1), self.enc(x10).flatten(1)], dim=1)).squeeze()

class Slot(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.slot = nn.Parameter(torch.randn(2, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x0, x5, x10):
        h0 = self.enc(x0).mean(dim=[2, 3]).unsqueeze(1) + self.slot.unsqueeze(0)
        return self.predict(h0[:, 0]).squeeze()

# Train Baseline
print("\nTraining Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(8):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_b = F.mse_loss(m(X0, X5, X10), Y).item()

# Train Slot
print("Training Slot...")
m = Slot()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(8):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_s = F.mse_loss(m(X0, X5, X10), Y).item()

random_mse = Y.var().item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Baseline: {mse_b:.4f}")
print(f"Slot:    {mse_s:.4f}")
print(f"Random:  {random_mse:.4f}")
print(f"\nSlot improvement: {(mse_b - mse_s) / mse_b * 100:.1f}%")
