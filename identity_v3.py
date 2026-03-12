"""
Identity Test - FINAL VERSION (Simplified)
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
print("IDENTITY TEST - FINAL")
print("="*60)

# Generate sequences
def generate_data(n=2000):
    """Generate sequences: t0 (colored) -> t5 (crossed) -> t10 (target)"""
    all_t0 = []  # First frame (colored)
    all_target = []  # Target position
    
    for _ in range(n):
        # Ball A: red, bounces
        x_a, y_a = 10.0, 16.0
        vx_a, vy_a = 1.5, 0.0
        
        # Ball B: blue, wraps
        x_b, y_b = 22.0, 16.0
        vx_b, vy_b = -1.5, 0.0
        
        # t0: colored
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[int(y_a), int(x_a)] = [1, 0, 0]  # Red
        img[int(y_b), int(x_b)] = [0, 0, 1]  # Blue
        all_t0.append(img)
        
        # Simulate to t10
        for _ in range(10):
            # Ball A bounces
            x_a, y_a = x_a + vx_a * 0.8, y_a + vy_a * 0.8
            if x_a < 3 or x_a > 29: vx_a *= -1
            if y_a < 3 or y_a > 29: vy_a *= -1
            
            # Ball B wraps
            x_b, y_b = x_b + vx_b * 0.8, y_b + vy_b * 0.8
            x_b = (x_b - 3) % 26 + 3
            y_b = (y_b - 3) % 26 + 3
        
        # Target: ball A position
        all_target.append([x_a / 32, y_a / 32])
    
    return np.array(all_t0), np.array(all_target)

print("\n1. Generating data...")
X, Y = generate_data(2000)
X = torch.FloatTensor(X).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)
print(f"   X: {X.shape}, Y: {Y.shape}")

# Models
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
        self.slot = nn.Parameter(torch.randn(2, 32) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        
    def forward(self, x):
        h = self.enc(x).mean(dim=[2,3])
        h = h.unsqueeze(1) + self.slot.unsqueeze(0)
        return self.predict(h[:, 0])

# Train
print("\n2. Training...")

results = {}

# Baseline
print("   Baseline:")
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
print("   Slot:")
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
