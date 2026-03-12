"""
EXPERIMENT 1: Partial Observability
Key: Objects move behind wall (occluded)

Hypothesis: When objects are occluded, object-level representation should help
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
print("EXPERIMENT 1: PARTIAL OBSERVABILITY")
print("="*60)

def generate_data(n=2000):
    """Objects move behind wall - occluded for several frames"""
    X_visible, X_occluded, X_reappear = [], [], []
    targets = []
    
    for _ in range(n):
        # Two balls at different positions
        x1 = random.uniform(5, 15)
        x2 = random.uniform(17, 27)
        y = random.uniform(10, 22)
        
        v1 = random.uniform(-2, 2)
        v2 = random.uniform(-2, 2)
        
        # t0: both visible
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x1)] = [1, 0, 0]  # Red = ball 1
        img0[int(y), int(x2)] = [0, 0, 1]  # Blue = ball 2
        
        # Move balls to wall position (behind wall)
        wall_x = 16  # Center of wall
        
        # Ball 1 goes behind wall
        # Ball 2 stays visible
        for step in range(5):
            x1 += v1 * 0.5
            x2 += v2 * 0.5
            if x1 < 3 or x1 > 29: v1 *= -1
            if x2 < 3 or x2 > 29: v2 *= -1
        
        # t5: occluded state
        # Ball 1 behind wall, ball 2 visible
        img5 = np.zeros((32, 32, 3), np.float32)
        
        # Wall covers middle section
        wall_color = [0.3, 0.3, 0.3]
        for wx in range(12, 20):
            for wy in range(5, 27):
                img5[wy, wx] = wall_color
        
        # Ball 2 still visible
        if 3 <= int(x2) < 32:
            img5[int(y), int(x2)] = [0, 0, 1]
        
        # Continue moving
        for step in range(5):
            x1 += v1 * 0.5
            x2 += v2 * 0.5
            if x1 < 3 or x1 > 29: v1 *= -1
            if x2 < 3 or x2 > 29: v2 *= -1
        
        # t10: reappear
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x1)] = [1, 1, 1]  # Both white
        img10[int(y), int(x2)] = [1, 1, 1]
        
        X_visible.append(img0)
        X_occluded.append(img5)
        X_reappear.append(img10)
        # Target: position of ball 1 (the occluded one)
        targets.append(x1 / 32)
    
    return np.array(X_visible), np.array(X_occluded), np.array(X_reappear), np.array(targets)

print("\n1. Generating occlusion data...")
X0, X5, X10, Y = generate_data(2000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   X0: {X0.shape}, X5: {X5.shape}, X10: {X10.shape}")

# Models
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8*3, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x0, x5, x10):
        h0 = self.enc(x0).flatten(1)
        h5 = self.enc(x5).flatten(1)
        h10 = self.enc(x10).flatten(1)
        return self.fc(torch.cat([h0, h5, h10], dim=1)).squeeze()

class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slots = nn.Parameter(torch.randn(2, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x0, x5, x10):
        h = self.enc(x10).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

print("\n2. Training Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_baseline = F.mse_loss(m(X0, X5, X10), Y).item()

print("3. Training Slot...")
m = SlotModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_slot = F.mse_loss(m(X0, X5, X10), Y).item()

random_mse = Y.var().item()

print("\n" + "="*60)
print("RESULTS - PARTIAL OBSERVABILITY")
print("="*60)
print(f"Baseline: MSE = {mse_baseline:.4f} ({(random_mse-mse_baseline)/random_mse*100:.1f}% < random)")
print(f"Slot:     MSE = {mse_slot:.4f} ({(random_mse-mse_slot)/random_mse*100:.1f}% < random)")
print(f"Random:   MSE = {random_mse:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse_slot < mse_baseline:
    print("=> Slot BETTER with occlusion!")
    print("=> Object-level representation helps with memory")
else:
    print("=> Baseline still wins")
    print("=> Need stronger occlusion or different task")
