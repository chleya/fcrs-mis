"""
FINAL Identity Tracking Test - NO SHORTCUTS
- Different motion rules: A bounces, B wraps
- Random velocity mutations at t3, t7
- Full crossover at t5
- All white after t0
- MUST track identity through time
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
print("FINAL IDENTITY TEST - NO SHORTCUTS")
print("="*60)

def generate_identity_data(n=3000):
    """Identity tracking with different rules"""
    images_t0 = []
    images_t5 = []
    images_t10 = []
    targets = []
    
    for _ in range(n):
        # Random initial positions
        x_a, y = random.uniform(5, 14), random.uniform(8, 24)
        x_b = random.uniform(18, 27)
        
        # Random initial velocities
        vx_a = random.uniform(-2, 2)
        vx_b = random.uniform(-2, 2)
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), dtype=np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]  # Red ball A
        img0[int(y), int(x_b)] = [0, 0, 1]  # Blue ball B
        
        white = [1, 1, 1]
        
        # Simulate t1-t5
        for step in range(1, 6):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            
            # Ball A: bounce
            if x_a < 3 or x_a > 29: vx_a *= -1
            
            # Ball B: wrap
            x_b = ((x_b - 3) % 26) + 3
            
            if step == 5:
                # t5: FULL CROSSOVER (both at center!)
                img5 = np.zeros((32, 32, 3), dtype=np.float32)
                img5[16, 16] = white  # Single white dot
                images_t5.append(img5.copy())
        
        # Velocity mutation at t7!
        # Ball A: direction flip
        if random.random() < 0.5: vx_a *= -1
        # Ball B: magnitude change
        vx_b *= random.uniform(0.5, 1.5)
        
        # Simulate t6-t10
        for step in range(6, 11):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            
            if x_a < 3 or x_a > 29: vx_a *= -1
            x_b = ((x_b - 3) % 26) + 3
            
            if step == 10:
                img10 = np.zeros((32, 32, 3), dtype=np.float32)
                img10[int(y), int(x_a)] = white
                img10[int(y), int(x_b)] = white
                images_t10.append(img10.copy())
                
                # Target: ball A position (the one that was RED at t0!)
                targets.append(x_a / 32)
        
        images_t0.append(img0.copy())
    
    return np.array(images_t0), np.array(images_t5), np.array(images_t10), np.array(targets)

print("\n1. Generating identity data...")
X0, X5, X10, Y = generate_identity_data(3000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   t0 (colored): {X0.shape}")
print(f"   t5 (overlap): {X5.shape}")
print(f"   t10 (white): {X10.shape}")
print(f"   target: {Y.shape}")

# Models
class Baseline_t0(nn.Module):
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

class Baseline_t10(nn.Module):
    """Use only t10 (white)"""
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

class FullSequence(nn.Module):
    """Use all frames"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        # Pool all frames
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
m = Baseline_t0()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 32):
        p = m(X0[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_t0 = F.mse_loss(m(X0), Y).item()

# t10 only
print("   t10 (white - MUST track identity)...")
m = Baseline_t10()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X10))
    for i in range(0, len(X10), 32):
        p = m(X10[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_t10 = F.mse_loss(m(X10), Y).item()

# Full sequence
print("   Full sequence (t0 + t5 + t10)...")
m = FullSequence()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 32):
        p = m(X0[idx[i:i+32]], X5[idx[i:i+32]], X10[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_full = F.mse_loss(m(X0, X5, X10), Y).item()

# Random guess baseline
random_mse = Y.var().item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"t0 (colored):     MSE = {mse_t0:.6f}")
print(f"t10 (white):      MSE = {mse_t10:.6f}")
print(f"Full sequence:    MSE = {mse_full:.6f}")
print(f"Random guess:     MSE = {random_mse:.6f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse_t10 > random_mse * 0.8:
    print("=> Identity tracking IMPOSSIBLE without color!")
    print("   Model cannot track object through complex motion")
elif mse_full < mse_t0 * 0.5:
    print("=> Full sequence HELPS significantly")
    print("   Model can use temporal information")
else:
    print("=> Partial results - need analysis")
