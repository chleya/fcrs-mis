"""
TEMPORAL IDENTITY BINDING TEST
Key difference: Can model track object identity through TIME?

Task: 
- t0: red ball at position A, blue ball at position B
- t5: both white balls (NO color info!)
- t10: predict where the ORIGINAL red ball is

This is the REAL identity tracking test - NOT spatial binding
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
print("TEMPORAL IDENTITY BINDING TEST")
print("="*60)

def generate_temporal_data(n=3000):
    """Temporal identity tracking"""
    images_t0 = []
    images_t5 = []
    targets = []
    
    for _ in range(n):
        # t0: colored balls
        x_red = random.uniform(4, 14)
        x_blue = random.uniform(18, 28)
        y = random.uniform(8, 24)
        
        v_red = random.uniform(-2, 2)
        v_blue = random.uniform(-2, 2)
        
        # Draw t0
        img0 = np.zeros((32, 32, 3), dtype=np.float32)
        img0[int(y), int(x_red)] = [1, 0, 0]  # Red
        img0[int(y), int(x_blue)] = [0, 0, 1]  # Blue
        
        # Simulate to t5 (white balls)
        for _ in range(5):
            x_red += v_red * 0.5
            x_blue += v_blue * 0.5
            if x_red < 3 or x_red > 29: v_red *= -1
            if x_blue < 3 or x_blue > 29: v_blue *= -1
        
        # Draw t5 (white)
        img5 = np.zeros((32, 32, 3), dtype=np.float32)
        img5[int(y), int(x_red)] = [1, 1, 1]
        img5[int(y), int(x_blue)] = [1, 1, 1]
        
        # Target: where is original RED ball?
        target = x_red / 32
        
        images_t0.append(img0)
        images_t5.append(img5)
        targets.append(target)
    
    return np.array(images_t0), np.array(images_t5), np.array(targets)

print("\n1. Generating temporal data...")
X0, X5, Y = generate_temporal_data(3000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   t0 (colored): {X0.shape}")
print(f"   t5 (white): {X5.shape}")
print(f"   target: {Y.shape}")

# Models
class Baseline_t0(nn.Module):
    """Use only t0 (has color info)"""
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

class Baseline_t5(nn.Module):
    """Use only t5 (NO color - must track identity)"""
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

class TwoFrame(nn.Module):
    """Use both t0 and t5"""
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
    def forward(self, x0, x5):
        h0 = self.enc(x0).flatten(1)
        h5 = self.enc(x5).flatten(1)
        return self.fc(torch.cat([h0, h5], dim=1)).squeeze(-1)

print("\n2. Training...")

# t0 only (has color)
print("   Baseline_t0 (colored)...")
m = Baseline_t0()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 32):
        p = m(X0[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_t0 = F.mse_loss(m(X0), Y).item()

# t5 only (white - identity required!)
print("   Baseline_t5 (white - identity tracking)...")
m = Baseline_t5()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X5))
    for i in range(0, len(X5), 32):
        p = m(X5[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_t5 = F.mse_loss(m(X5), Y).item()

# Two frame
print("   TwoFrame (t0 + t5)...")
m = TwoFrame()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 32):
        p = m(X0[idx[i:i+32]], X5[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_two = F.mse_loss(m(X0, X5), Y).item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"t0 (colored):  MSE = {mse_t0:.6f}")
print(f"t5 (white):    MSE = {mse_t5:.6f}")
print(f"t0 + t5:      MSE = {mse_two:.6f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print(f"Color info helps: {(mse_t5 - mse_t0) / mse_t5 * 100:.1f}% reduction")
print(f"Two frames: {mse_two:.4f}")

if mse_t5 > mse_t0 * 3:
    print("\n=> Identity tracking FAILS without color!")
    print("   Model cannot track object through time")
else:
    print("\n=> Model CAN track identity through time")
