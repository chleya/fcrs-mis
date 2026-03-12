"""
HARD Identity Test - NO MOTION INFERENCE POSSIBLE
- t0: colored balls (red A, blue B)  
- t1-t4: white balls moving normally
- t5: RANDOM TELEPORTATION - objects appear at completely RANDOM positions!
- t6-t9: white balls moving
- t10: predict ball A

KEY: At t5, positions are RANDOM - no motion continuity!
Only way to know identity is to track from t0.
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
print("HARD IDENTITY TEST - RANDOM TELEPORTATION")
print("="*60)

def generate_data(n=3000):
    X_t0 = []
    X_t5 = []
    X_t10 = []
    Y = []
    
    for _ in range(n):
        # t0: initial colored positions
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        vx_a = random.uniform(-2, 2)
        vx_b = random.uniform(-2, 2)
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]  # Red
        img0[int(y), int(x_b)] = [0, 0, 1]  # Blue
        
        white = [1, 1, 1]
        
        # t1-t4: normal motion
        for _ in range(1, 5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
            if x_b < 3 or x_b > 29: vx_b *= -1
            x_b = ((x_b - 3) % 26) + 3
        
        # t5: RANDOM TELEPORTATION - completely random positions!
        # This BREAKS all motion inference
        x_a_t5 = random.uniform(5, 27)
        x_b_t5 = random.uniform(5, 27)
        
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a_t5)] = white
        img5[int(y), int(x_b_t5)] = white
        
        # t6-t9: new random velocities after teleport
        vx_a = random.uniform(-2, 2)
        vx_b = random.uniform(-2, 2)
        
        for _ in range(6, 10):
            x_a_t5 += vx_a * 0.5
            x_b_t5 += vx_b * 0.5
            if x_a_t5 < 3 or x_a_t5 > 29: vx_a *= -1
            if x_b_t5 < 3 or x_b_t5 > 29: vx_b *= -1
        
        # t10: final positions
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x_a_t5)] = white
        img10[int(y), int(x_b_t5)] = white
        
        X_t0.append(img0)
        X_t5.append(img5)
        X_t10.append(img10)
        Y.append(x_a_t5 / 32)  # Target: original A (now at random pos)
    
    return np.array(X_t0), np.array(X_t5), np.array(X_t10), np.array(Y)

print("\n1. Generating data...")
X0, X5, X10, Y = generate_data(3000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   X0: {X0.shape}, X5: {X5.shape}, X10: {X10.shape}")
print(f"   Y: {Y.shape}, var: {Y.var():.4f}")

# Verify teleportation broke correlation
corr_t0 = np.corrcoef(X0[:, 0, 16, :].argmax(axis=1), Y)[0, 1]
corr_t5 = np.corrcoef(X5[:, 0, 16, :].argmax(axis=1), Y)[0, 1]
print(f"\n   Correlation with target:")
print(f"   t0 position -> target: {corr_t0:.3f}")
print(f"   t5 position -> target: {corr_t5:.3f}")

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
print("   t0 (colored - has identity marker)...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 32):
        p = m(X0[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_t0 = F.mse_loss(m(X0), Y).item()

# t5 only (teleported - no identity info!)
print("   t5 (teleported - random positions)...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X5))
    for i in range(0, len(X5), 32):
        p = m(X5[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_t5 = F.mse_loss(m(X5), Y).item()

# t10 only
print("   t10 (final white)...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
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
for ep in range(15):
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
print(f"t5 (teleport):    MSE = {mse_t5:.6f}")
print(f"t10 (final):      MSE = {mse_t10:.6f}")
print(f"Full sequence:    MSE = {mse_full:.6f}")
print(f"Random guess:      MSE = {random_mse:.6f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print(f"t0 vs random: {(random_mse - mse_t0) / random_mse * 100:.1f}% better than random")
print(f"Full vs random: {(random_mse - mse_full) / random_mse * 100:.1f}% better than random")

if mse_full < random_mse * 0.5:
    print("\n=> Model CAN track identity through teleportation!")
    print("   Must be using t0 color info")
elif mse_t0 < mse_full:
    print("\n=> t0 (colored) is most important")
    print("   Without color, identity is lost")
else:
    print("\n=> Need more analysis")
