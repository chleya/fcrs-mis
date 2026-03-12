"""
EXPERIMENT 2: Object Count Change
Key: Objects appear/disappear - trajectory can't track

This tests if object representations emerge when count changes
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

def generate_data(n=2000):
    """Object count changes mid-sequence"""
    X_t0, X_t5, X_t10 = [], [], []
    targets = []
    
    for _ in range(n):
        # Random number of initial objects (2-4)
        n_init = random.randint(2, 4)
        
        # Initial positions
        positions = []
        velocities = []
        for i in range(n_init):
            x = random.uniform(5, 27)
            y = random.uniform(8, 24)
            positions.append((x, y))
            velocities.append((random.uniform(-2, 2), random.uniform(-2, 2)))
        
        # t0: colored balls
        img0 = np.zeros((32, 32, 3), np.float32)
        for i in range(n_init):
            x, y = positions[i]
            if i == 0:
                img0[int(y), int(x)] = [1, 0, 0]  # Red = target
            else:
                img0[int(y), int(x)] = [random.random(), random.random(), random.random()]
        
        # Move to t5 - reduce count (one disappears)
        for step in range(5):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x += vx * 0.5
                y += vy * 0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i] = (x, y)
                velocities[i] = (vx, vy)
        
        # Randomly remove one object at t5
        if len(positions) > 1:
            remove_idx = random.randint(0, len(positions)-1)
            positions = positions[:remove_idx] + positions[remove_idx+1:]
            velocities = velocities[:remove_idx] + velocities[remove_idx+1:]
        
        # t5: fewer objects
        img5 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img5[int(y), int(x)] = [1, 1, 1]
        
        # Move to t10
        for step in range(5):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x += vx * 0.5
                y += vy * 0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i] = (x, y)
        
        # t10: final
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img10[int(y), int(x)] = [1, 1, 1]
        
        X_t0.append(img0)
        X_t5.append(img5)
        X_t10.append(img10)
        
        # Target: position of original ball 0 (if still exists)
        if len(positions) > 0:
            targets.append(positions[0][0] / 32)
        else:
            targets.append(0.5)
    
    return np.array(X_t0), np.array(X_t5), np.array(X_t10), np.array(targets)

print("\n1. Generating object count change data...")
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
    def __init__(self, n_slots=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slots = nn.Parameter(torch.randn(n_slots, 64) * 0.1)
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
m = SlotModel(n_slots=4)
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
print("RESULTS - OBJECT COUNT CHANGE")
print("="*60)
print(f"Baseline: MSE = {mse_baseline:.4f} ({(random_mse-mse_baseline)/random_mse*100:.1f}% < random)")
print(f"Slot:     MSE = {mse_slot:.4f} ({(random_mse-mse_slot)/random_mse*100:.1f}% < random)")
print(f"Random:   MSE = {random_mse:.4f}")
