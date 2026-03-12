"""
EXPERIMENT 2: Object Count Change
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
print("EXPERIMENT 2: OBJECT COUNT CHANGE")
print("="*60)

def clamp(val):
    return max(3, min(28, int(val)))

def generate_data(n=2000):
    X0, X5, X10, Y = [], [], [], []
    
    for _ in range(n):
        n_init = random.randint(2, 3)
        positions = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_init)]
        velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_init)]
        
        # t0
        img0 = np.zeros((32, 32, 3), np.float32)
        for i, (x, y) in enumerate(positions):
            if i == 0:
                img0[clamp(y), clamp(x)] = [1, 0, 0]
            else:
                img0[clamp(y), clamp(x)] = [0, 0, 1]
        
        # Move 5 steps
        for _ in range(5):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx * 0.5, y + vy * 0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i] = (x, y)
                velocities[i] = (vx, vy)
        
        # Remove one
        if len(positions) > 1:
            positions = positions[1:]
            velocities = velocities[1:]
        
        # t5
        img5 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img5[clamp(y), clamp(x)] = [1, 1, 1]
        
        # Move 5 more
        for _ in range(5):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx * 0.5, y + vy * 0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i] = (x, y)
        
        # t10
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img10[clamp(y), clamp(x)] = [1, 1, 1]
        
        X0.append(img0)
        X5.append(img5)
        X10.append(img10)
        Y.append(positions[0][0] / 32 if positions else 0.5)
    
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

class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.slots = nn.Parameter(torch.randn(3, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x0, x5, x10):
        h = self.enc(x10).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

print("\nTraining Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_baseline = F.mse_loss(m(X0, X5, X10), Y).item()

print("Training Slot...")
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

print(f"\nBaseline: MSE={mse_baseline:.4f} ({(random_mse-mse_baseline)/random_mse*100:.1f}%)")
print(f"Slot: MSE={mse_slot:.4f} ({(random_mse-mse_slot)/random_mse*100:.1f}%)")
print(f"Random: MSE={random_mse:.4f}")
