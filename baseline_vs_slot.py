"""
COMPARISON: Baseline vs Slot with HARD identity task
This is THE critical test of "structure → capability" theory

If Slot (with object structure) outperforms Baseline significantly,
it validates that structure constraints can cause capability emergence.
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
print("BASELINE vs SLOT - IDENTITY TRACKING")
print("="*60)

def generate_data(n=3000):
    X_t0 = []
    X_t5 = []
    X_t10 = []
    Y = []
    
    for _ in range(n):
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        vx_a = random.uniform(-2, 2)
        vx_b = random.uniform(-2, 2)
        
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]
        img0[int(y), int(x_b)] = [0, 0, 1]
        
        white = [1, 1, 1]
        
        for _ in range(1, 5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
            if x_b < 3 or x_b > 29: vx_b *= -1
        
        # Random teleportation
        x_a_t5 = random.uniform(5, 27)
        x_b_t5 = random.uniform(5, 27)
        
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a_t5)] = white
        img5[int(y), int(x_b_t5)] = white
        
        vx_a = random.uniform(-2, 2)
        vx_b = random.uniform(-2, 2)
        
        for _ in range(6, 10):
            x_a_t5 += vx_a * 0.5
            x_b_t5 += vx_b * 0.5
            if x_a_t5 < 3 or x_a_t5 > 29: vx_a *= -1
        
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x_a_t5)] = white
        img10[int(y), int(x_b_t5)] = white
        
        X_t0.append(img0)
        X_t5.append(img5)
        X_t10.append(img10)
        Y.append(x_a_t5 / 32)
    
    return np.array(X_t0), np.array(X_t5), np.array(X_t10), np.array(Y)

print("\n1. Generating data...")
X0, X5, X10, Y = generate_data(3000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   Data: {X0.shape}")

# Baseline: standard CNN
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
        return self.fc(torch.cat([h0, h5, h10], dim=1)).squeeze(-1)

# Slot model: with object structure
class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        # Two slots for two objects
        self.slot = nn.Parameter(torch.randn(2, 64) * 0.1)
        # Predict position from each slot
        self.predict = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x0, x5, x10):
        # Process each frame
        h0 = self.enc(x0).mean(dim=[2, 3])  # (B, 64)
        h5 = self.enc(x5).mean(dim=[2, 3])
        h10 = self.enc(x10).mean(dim=[2, 3])
        
        # Add slots
        h0 = h0.unsqueeze(1) + self.slot.unsqueeze(0)  # (B, 2, 64)
        h5 = h5.unsqueeze(1) + self.slot.unsqueeze(0)
        h10 = h10.unsqueeze(1) + self.slot.unsqueeze(0)
        
        # Get predictions from slot 0 (should track ball A)
        p0 = self.predict(h0[:, 0])
        p5 = self.predict(h5[:, 0])
        p10 = self.predict(h10[:, 0])
        
        # Average across time
        return ((p0 + p5 + p10) / 3).squeeze(-1)

print("\n2. Training multiple seeds...")

random_mse = Y.var().item()
print(f"   Random MSE: {random_mse:.4f}")

results_baseline = []
results_slot = []

for seed in range(3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Baseline
    m = Baseline()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(15):
        idx = torch.randperm(len(X0))
        for i in range(0, len(X0), 32):
            p = m(X0[idx[i:i+32]], X5[idx[i:i+32]], X10[idx[i:i+32]])
            loss = F.mse_loss(p, Y[idx[i:i+32]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse_b = F.mse_loss(m(X0, X5, X10), Y).item()
    results_baseline.append(mse_b)
    
    # Slot
    m = SlotModel()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(15):
        idx = torch.randperm(len(X0))
        for i in range(0, len(X0), 32):
            p = m(X0[idx[i:i+32]], X5[idx[i:i+32]], X10[idx[i:i+32]])
            loss = F.mse_loss(p, Y[idx[i:i+32]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse_s = F.mse_loss(m(X0, X5, X10), Y).item()
    results_slot.append(mse_s)
    
    print(f"   Seed {seed}: Baseline={mse_b:.4f}, Slot={mse_s:.4f}")

print("\n" + "="*60)
print("FINAL RESULTS (3 seeds)")
print("="*60)

mean_b = np.mean(results_baseline)
std_b = np.std(results_baseline)
mean_s = np.mean(results_slot)
std_s = np.std(results_slot)

print(f"Baseline: {mean_b:.4f} ± {std_b:.4f}")
print(f"Slot:    {mean_s:.4f} ± {std_s:.4f}")
print(f"Random:  {random_mse:.4f}")

improvement = (mean_b - mean_s) / mean_b * 100
print(f"\nSlot improvement over Baseline: {improvement:.1f}%")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if improvement > 20:
    print("=> Slot SIGNIFICANTLY outperforms Baseline!")
    print("=> Structure constraints DO cause capability emergence")
elif improvement > 5:
    print("=> Slot moderately better than Baseline")
    print("=> Some benefit from object structure")
else:
    print("=> No significant difference")
    print("=> Structure constraints not sufficient alone")
