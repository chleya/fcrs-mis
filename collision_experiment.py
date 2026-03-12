"""
INTERACTION EXPERIMENT - Collision/Momentum Exchange

Goal: Test if Slot becomes necessary when objects interact (not just move independently)

Key: 
- Objects collide and exchange momentum
- Trajectory tracking alone is NOT sufficient
- Must model object-object interaction
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
print("INTERACTION EXPERIMENT - COLLISION")
print("="*60)

def generate_collision_data(n_samples=2000):
    """Generate collision scenarios"""
    X_t0, X_t10 = [], []
    targets = []
    
    for _ in range(n_samples):
        # Two balls heading toward each other
        x1 = random.uniform(4, 12)
        x2 = random.uniform(20, 28)
        y = random.uniform(10, 22)
        
        v1 = random.uniform(1, 2)   # Heading right
        v2 = random.uniform(-2, -1)  # Heading left
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x1)] = [1, 0, 0]  # Red = ball 1 (we track this)
        img0[int(y), int(x2)] = [0, 0, 1]  # Blue = ball 2
        
        # Simulate collision at t5
        # Simple elastic collision model
        for step in range(5):
            x1 += v1 * 0.5
            x2 += v2 * 0.5
        
        # Collision!
        if abs(x1 - x2) < 3:
            # Exchange velocities (elastic collision)
            v1, v2 = v2, v1
        
        # After collision
        for step in range(5, 10):
            x1 += v1 * 0.5
            x2 += v2 * 0.5
            # Bounce off walls
            if x1 < 3 or x1 > 29: v1 *= -1
            if x2 < 3 or x2 > 29: v2 *= -1
        
        # Clamp
        x1 = max(3, min(28, x1))
        x2 = max(3, min(28, x2))
        
        # t10: final positions (white)
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x1)] = [1, 1, 1]
        img10[int(y), int(x2)] = [1, 1, 1]
        
        X_t0.append(img0)
        X_t10.append(img10)
        # Target: position of red ball after collision
        targets.append(x1 / 32)
    
    return np.array(X_t0), np.array(X_t10), np.array(targets)

print("\n1. Generating collision data...")
X0, X10, Y = generate_collision_data(2000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   Data: {X0.shape}")
print(f"   Target: {Y.shape}")

# Models
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8*2, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x0, x10):
        h0 = self.enc(x0).flatten(1)
        h10 = self.enc(x10).flatten(1)
        return self.fc(torch.cat([h0, h10], dim=1)).squeeze()

class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slots = nn.Parameter(torch.randn(2, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x0, x10):
        h = self.enc(x10).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        # Use slot 0 for prediction
        return self.predict(h[:, 0]).squeeze()

print("\n2. Training Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_baseline = F.mse_loss(m(X0, X10), Y).item()

print("3. Training SlotModel...")
m = SlotModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_slot = F.mse_loss(m(X0, X10), Y).item()

random_mse = Y.var().item()

print("\n" + "="*60)
print("RESULTS - COLLISION")
print("="*60)
print(f"Baseline: MSE = {mse_baseline:.4f} ({(random_mse-mse_baseline)/random_mse*100:.1f}% < random)")
print(f"Slot:    MSE = {mse_slot:.4f} ({(random_mse-mse_slot)/random_mse*100:.1f}% < random)")
print(f"Random:  MSE = {random_mse:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse_slot < mse_baseline:
    print("=> Slot BETTER with collision/interaction!")
    print("=> Object interaction requires object-level representation")
else:
    print("=> Baseline still wins")
    print("=> Even collision doesn't force object representation")
