"""
L3 VERIFICATION: Slot + Temporal = Object Individuation

Goal: Verify if combining slot competition + temporal binding produces object individuation

Design:
1. Slot attention for spatial grouping
2. Temporal binding for tracking across frames
3. Test if objects are individuated
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
print("L3 VERIFICATION: SLOT + TEMPORAL")
print("="*60)

def generate_data(n=2000):
    """Two balls, need to track both"""
    X0, X5, X10 = [], [], []
    Y_a, Y_b = [], []
    
    for _ in range(n):
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        vx_a, vx_b = random.uniform(-2, 2), random.uniform(-2, 2)
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]
        img0[int(y), int(x_b)] = [0, 0, 1]
        
        white = [1, 1, 1]
        
        # t5
        for _ in range(5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
        
        x_a = max(3, min(28, x_a))
        x_b = max(3, min(28, x_b))
        
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a)] = white
        img5[int(y), int(x_b)] = white
        
        # t10
        for _ in range(5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
        
        x_a = max(3, min(28, x_a))
        x_b = max(3, min(28, x_b))
        
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x_a)] = white
        img10[int(y), int(x_b)] = white
        
        X0.append(img0)
        X5.append(img5)
        X10.append(img10)
        Y_a.append(x_a / 32)
        Y_b.append(x_b / 32)
    
    return np.array(X0), np.array(X5), np.array(X10), np.array(Y_a), np.array(Y_b)

print("\n1. Generating data...")
X0, X5, X10, Y_a, Y_b = generate_data(2000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y_a = torch.FloatTensor(Y_a)
Y_b = torch.FloatTensor(Y_b)

print(f"   Data: {X0.shape}")

# Model 1: Baseline
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8*3, 128), nn.ReLU(),
            nn.Linear(128, 2)  # Predict both A and B
        )
    
    def forward(self, x0, x5, x10):
        h0 = self.enc(x0).flatten(1)
        h5 = self.enc(x5).flatten(1)
        h10 = self.enc(x10).flatten(1)
        return self.fc(torch.cat([h0, h5, h10], dim=1))

# Model 2: Slot + Temporal
class SlotTemporal(nn.Module):
    """
    Slot attention + temporal binding for object individuation
    """
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        
        # Two slots
        self.slot = nn.Parameter(torch.randn(2, 64) * 0.1)
        
        # Dynamics for temporal binding
        self.dynamics = nn.Linear(64, 64)
        
        # Predict both positions
        self.predict = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x0, x5, x10):
        # Encode frames
        f0 = self.enc(x0).mean(dim=[2, 3])
        f5 = self.enc(x5).mean(dim=[2, 3])
        f10 = self.enc(x10).mean(dim=[2, 3])
        
        # Add slots
        h0 = f0.unsqueeze(1) + self.slot.unsqueeze(0)  # (B, 2, 64)
        h5 = f5.unsqueeze(1) + self.slot.unsqueeze(0)
        h10 = f10.unsqueeze(1) + self.slot.unsqueeze(0)
        
        # Temporal binding loss
        temp_loss = 0
        for h_t, h_t1 in [(h0, h5), (h5, h10)]:
            pred = self.dynamics(h_t)
            temp_loss += F.mse_loss(pred, h_t1)
        
        # Predict from both slots
        p0 = self.predict(h10[:, 0])  # Slot 0
        p1 = self.predict(h10[:, 1])  # Slot 1
        
        return torch.cat([p0, p1], dim=1), temp_loss

print("\n2. Training Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, torch.stack([Y_a[idx[i:i+64]], Y_b[idx[i:i+64]]], dim=1))
        opt.zero_grad(); loss.backward(); opt.step()

pred_b = m(X0, X5, X10)
mse_baseline = (F.mse_loss(pred_b[:, 0], Y_a) + F.mse_loss(pred_b[:, 1], Y_b)).item() / 2

print("3. Training Slot+Temporal...")
m = SlotTemporal()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p, temp_loss = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        
        # Prediction loss
        loss_main = (F.mse_loss(p[:, 0], Y_a[idx[i:i+64]]) + 
                     F.mse_loss(p[:, 1], Y_b[idx[i:i+64]])) / 2
        
        # Temporal binding loss
        loss = loss_main + 0.1 * temp_loss
        opt.zero_grad(); loss.backward(); opt.step()

pred_s, temp_loss = m(X0, X5, X10)
mse_slot = (F.mse_loss(pred_s[:, 0], Y_a) + F.mse_loss(pred_s[:, 1], Y_b)).item() / 2

random_mse = (Y_a.var() + Y_b.var()).item() / 2

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Baseline:        MSE = {mse_baseline:.4f} ({(random_mse-mse_baseline)/random_mse*100:.1f}% < random)")
print(f"Slot+Temporal:   MSE = {mse_slot:.4f} ({(random_mse-mse_slot)/random_mse*100:.1f}% < random)")
print(f"Random:         MSE = {random_mse:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse_slot < mse_baseline:
    print("=> Slot+Temporal BETTER than Baseline")
    print("=> Object individuation may have emerged")
else:
    print("=> No improvement from Slot+Temporal")
    print("=> Need stronger structure or different task")
