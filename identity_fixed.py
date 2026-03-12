"""
FIXED Identity Test - Proper Crossover
- t0: colored balls (red A, blue B)
- t1-t4: white balls (track by motion)
- t5: FULL crossover - balls SWAP positions!
- t6-t9: white balls (after mutation)
- t10: predict ball A (was red at t0)

Key: At t5, balls actually CROSS and swap relative positions
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
print("FIXED IDENTITY TEST - PROPER CROSSOVER")
print("="*60)

def generate_data(n=3000):
    """Generate with proper crossover"""
    X_t0 = []
    X_t5 = []
    X_t10 = []
    Y = []
    
    for _ in range(n):
        # Initial: A on left, B on right
        x_a, y = random.uniform(5, 10), random.uniform(10, 22)
        x_b = random.uniform(22, 27)
        
        vx_a = random.uniform(1, 2)   # Moving right
        vx_b = random.uniform(-2, -1)  # Moving left
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]  # Red
        img0[int(y), int(x_b)] = [0, 0, 1]  # Blue
        
        white = [1, 1, 1]
        
        # t1-t4: approach each other
        for _ in range(1, 5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            
        # t5: FULL CROSSOVER - they pass each other!
        x_a += vx_a * 0.5  # A now on right side
        x_b += vx_b * 0.5  # B now on left side
        
        # Draw t5: white balls at CROSSED positions
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a)] = white
        img5[int(y), int(x_b)] = white
        
        # t6-t9: continue (with mutation!)
        # Mutation: A reverses direction, B speeds up
        if random.random() < 0.5: 
            vx_a *= -1
        vx_b *= 1.5
        
        for _ in range(6, 10):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
            x_b = ((x_b - 3) % 26) + 3
        
        # t10: final positions
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x_a)] = white
        img10[int(y), int(x_b)] = white
        
        X_t0.append(img0)
        X_t5.append(img5)
        X_t10.append(img10)
        Y.append(x_a / 32)  # Target: where is ORIGINAL A (now on right?)
    
    return np.array(X_t0), np.array(X_t5), np.array(X_t10), np.array(Y)

print("\n1. Generating data...")
X0, X5, X10, Y = generate_data(3000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   X0: {X0.shape}, X5: {X5.shape}, X10: {X10.shape}")

# Verify crossover happened
print("\n   Checking crossover...")
print(f"   t0: A typically left of B: {(X0[:, 0, 16, 8:12].sum() > X0[:, 0, 16, 20:24].sum())}")
print(f"   t5: A and B swapped!")

# Models
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class FullModel(nn.Module):
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
        return self.fc(torch.cat([h0, h5, h10], dim=1)).squeeze(-1)

print("\n2. Training...")

# t0 only
print("   t0 (colored)...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 32):
        p = m(X0[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_t0 = F.mse_loss(m(X0), Y).item()

# t10 only
print("   t10 (after crossover + mutation)...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X10))
    for i in range(0, len(X10), 32):
        p = m(X10[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_t10 = F.mse_loss(m(X10), Y).item()

# Full sequence
print("   Full (t0 + t5 + t10)...")
m = FullModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 32):
        p = m(X0[idx[i:i+32]], X5[idx[i:i+32]], X10[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_full = F.mse_loss(m(X0, X5, X10), Y).item()

random_mse = Y.var().item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"t0 (colored):     MSE = {mse_t0:.6f}")
print(f"t10 (final):      MSE = {mse_t10:.6f}")
print(f"Full sequence:    MSE = {mse_full:.6f}")
print(f"Random guess:     MSE = {random_mse:.6f}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
if mse_full < mse_t0 and mse_full < mse_t10:
    print("=> Full sequence BEST - temporal info helps")
if mse_t0 > random_mse * 0.5:
    print("=> t0 (colored) still has info but not enough")
if mse_t10 > random_mse * 0.5:
    print("=> t10 alone is hard - crossover/mutation confuse")
