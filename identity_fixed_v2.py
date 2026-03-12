"""Identity Test - FIXED v2"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("="*60)
print("IDENTITY TEST - FIXED v2")
print("="*60)

# Dataset
def create_dataset(n=6000, crossing_ratio=0.3):
    data = []
    n_cross = int(n * crossing_ratio)
    
    for i in range(n):
        t = i * 0.015
        mode = 2 if i < n_cross else 0
        
        if mode == 2:  # Crossing
            ball_a_x = 5.0 + t * 10
            ball_b_x = 27.0 - t * 10
        else:  # No crossing
            ball_a_x = 8.0 + t * 0.5
            ball_b_x = 24.0 - t * 0.5
        
        ball_y = 16.0
        
        # Clamp
        ball_a_x = max(2, min(30, ball_a_x))
        ball_b_x = max(2, min(30, ball_b_x))
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[int(ball_y), int(ball_a_x)] = [1.0, 1.0, 1.0]
        img[int(ball_y), int(ball_b_x)] = [0.5, 0.5, 0.5]
        
        target = [ball_a_x / 32, ball_y / 32]
        data.append((img, target, mode))
    
    return data

# Simple Slot with FIXED position embedding
class SlotFixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        
        # Fixed position embeddings - slot 0 = left, slot 1 = right
        self.slot_embed = nn.Parameter(torch.randn(2, 64) * 0.1)
        
        # Per-slot predictors (FIXED binding!)
        self.pred_a = nn.Linear(64, 2)  # Ball A
        self.pred_b = nn.Linear(64, 2)  # Ball B
        
    def forward(self, x):
        h = self.encoder(x)  # (B, 64, 8, 8)
        h = h.mean(dim=[2,3])  # (B, 64)
        
        # Add fixed position embedding to features
        h = h.unsqueeze(1) + self.slot_embed.unsqueeze(0)  # (B, 2, 64)
        
        # Slot 0 predicts ball A, slot 1 predicts ball B
        pred_a = self.pred_a(h[:, 0, :])  # FIXED to slot 0
        pred_b = self.pred_b(h[:, 1, :])  # FIXED to slot 1
        
        return pred_a, pred_b

# Baseline
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# Training with progressive difficulty
# ============================================================

print("\n1. Creating datasets...")

# Easy: no crossing
data_easy = create_dataset(2000, crossing_ratio=0.0)
X_easy = torch.FloatTensor(np.array([d[0] for d in data_easy])).permute(0,3,1,2)
Y_easy = torch.FloatTensor([d[1] for d in data_easy])

# Medium: 10%
data_med = create_dataset(2000, crossing_ratio=0.1)
X_med = torch.FloatTensor(np.array([d[0] for d in data_med])).permute(0,3,1,2)
Y_med = torch.FloatTensor([d[1] for d in data_med])

# Hard: 30%
data_hard = create_dataset(2000, crossing_ratio=0.3)
X_hard = torch.FloatTensor(np.array([d[0] for d in data_hard])).permute(0,3,1,2)
Y_hard = torch.FloatTensor([d[1] for d in data_hard])
modes = np.array([d[2] for d in data_hard])

print(f"   Easy: {len(X_easy)}, Medium: {len(X_med)}, Hard: {len(X_hard)}")

results = {}

# Train Baseline
print("\n2. Training Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

phases = [(X_easy, Y_easy, 3), (X_med, Y_med, 3), (X_hard, Y_hard, 6)]
for X, Y, iters in phases:
    for ep in range(iters):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            p = m(X[idx[i:i+32]])
            loss = F.mse_loss(p, Y[idx[i:i+32]])
            opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    p = m(X_hard)
    results['Baseline'] = {
        'cross': F.mse_loss(p[modes==2], Y_hard[modes==2]).item(),
        'normal': F.mse_loss(p[modes==0], Y_hard[modes==0]).item()
    }

# Train SlotFixed
print("\n3. Training SlotFixed...")
m = SlotFixed()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

phases = [(X_easy, Y_easy, 3), (X_med, Y_med, 3), (X_hard, Y_hard, 6)]
for X, Y, iters in phases:
    for ep in range(iters):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            pa, pb = m(X[idx[i:i+32]])
            
            # Main loss: predict ball A with slot 0 (FIXED!)
            loss = F.mse_loss(pa, Y[idx[i:i+32]])
            
            # Constraint: slot separation
            sep = F.mse_loss(pa, pb)
            loss = loss + 0.5 * sep
            
            opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    pa, pb = m(X_hard)
    # Use slot 0 (FIXED to ball A)
    results['SlotFixed'] = {
        'cross': F.mse_loss(pa[modes==2], Y_hard[modes==2]).item(),
        'normal': F.mse_loss(pa[modes==0], Y_hard[modes==0]).item()
    }

# Summary
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"{'Model':<15} | {'Cross':>10} | {'Normal':>10}")
print("-"*45)
for name, r in results.items():
    print(f"{name:<15} | {r['cross']:>10.4f} | {r['normal']:>10.4f}")

diff = results['Baseline']['cross'] - results['SlotFixed']['cross']
print(f"\nDifference: {diff:+.4f}")
if diff > 0:
    print("=> SlotFixed is BETTER!")
