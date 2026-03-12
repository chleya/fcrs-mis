"""
Identity Test - INTERMEDIATE FRAMES (identity MUST be tracked)
- t0: colored balls (red/blue)
- t1-t4: SAME color (identity required!)
- t5: overlap
- t10: predict
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
print("IDENTITY TEST - WITH INTERMEDIATE FRAMES")
print("="*60)

def generate_data(n=5000):
    all_t0, all_t4, all_target = [], [], []
    
    for _ in range(n):
        # Random setup
        x_a, y_a = random.uniform(5, 12), random.uniform(8, 24)
        x_b, y_b = random.uniform(20, 27), random.uniform(8, 24)
        
        vx_a, vy_a = random.uniform(-2, 2), random.uniform(-2, 2)
        vx_b, vy_b = random.uniform(-2, 2), random.uniform(-2, 2)
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), dtype=np.float32)
        img0[int(y_a), int(x_a)] = [1, 0, 0]  # Red
        img0[int(y_b), int(x_b)] = [0, 0, 1]  # Blue
        all_t0.append(img0)
        
        # t1-t4: SAME color (identity required!)
        white = [1, 1, 1]
        for step in range(4):
            x_a, y_a = x_a + vx_a * 0.8, y_a + vy_a * 0.8
            x_b, y_b = x_b + vx_b * 0.8, y_b + vy_b * 0.8
            
            # Bounce for A
            if x_a < 3 or x_a > 29: vx_a *= -1
            if y_a < 3 or y_a > 29: vy_a *= -1
            
            # Wrap for B
            x_b = ((x_b - 3) % 26) + 3
            y_b = ((y_b - 3) % 26) + 3
            
            if step == 3:  # Save t4 frame
                img4 = np.zeros((32, 32, 3), dtype=np.float32)
                img4[int(y_a), int(x_a)] = white
                img4[int(y_b), int(x_b)] = white
                all_t4.append(img4)
        
        # Simulate to t10
        for _ in range(6):
            x_a, y_a = x_a + vx_a * 0.8, y_a + vy_a * 0.8
            x_b, y_b = x_b + vx_b * 0.8, y_b + vy_b * 0.8
            if x_a < 3 or x_a > 29: vx_a *= -1
            if y_a < 3 or y_a > 29: vy_a *= -1
            x_b = ((x_b - 3) % 26) + 3
            y_b = ((y_b - 3) % 26) + 3
        
        all_target.append([x_a / 32, y_a / 32])
    
    return np.array(all_t0), np.array(all_t4), np.array(all_target)

print("\n1. Generating data...")
X0, X4, Y = generate_data(5000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X4 = torch.FloatTensor(X4).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   t0 (colored): {X0.shape}")
print(f"   t4 (white): {X4.shape}")
print(f"   target: {Y.shape}")

# Models
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.enc(x)

class TwoFrameModel(nn.Module):
    """Uses BOTH t0 and t4"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8*2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x0, x4):
        h0 = self.enc(x0).flatten(1)
        h4 = self.enc(x4).flatten(1)
        h = torch.cat([h0, h4], dim=1)
        return self.fc(h)

class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slot = nn.Parameter(torch.randn(2, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        
    def forward(self, x):
        h = self.enc(x).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slot.unsqueeze(0)
        return self.predict(h[:, 0])

print("\n2. Training...")

results = {}

# Baseline: only t0
print("   Baseline (t0 only)...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 32):
        p = m(X0[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
results['Baseline_t0'] = F.mse_loss(m(X0), Y).item()

# TwoFrame: t0 + t4
print("   TwoFrame (t0 + t4)...")
m = TwoFrameModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 32):
        p = m(X0[idx[i:i+32]], X4[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
results['TwoFrame'] = F.mse_loss(m(X0, X4), Y).item()

# Slot: uses t4 only (white balls - needs identity!)
print("   SlotModel (t4 only - white balls)...")
m = SlotModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X4))
    for i in range(0, len(X4), 32):
        p = m(X4[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
results['Slot_t4'] = F.mse_loss(m(X4), Y).item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
for name, mse in results.items():
    print(f"{name}: MSE = {mse:.4f}")
