"""Identity Test - RANDOM motion (真正hard)"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42); np.random.seed(42); torch.manual_seed(42)

print("="*60)
print("IDENTITY TEST - RANDOM MOTION")
print("="*60)

# Dataset with RANDOM motion (identity MUST be tracked)
def create_dataset(n=5000):
    data = []
    
    for i in range(n):
        # Random initial positions
        x1 = random.uniform(5, 15)
        x2 = random.uniform(17, 27)
        y = random.uniform(10, 22)
        
        # Random velocities
        v1x = random.uniform(-2, 2)
        v2x = random.uniform(-2, 2)
        
        # Ball A and B have SAME appearance
        # But we track which started on LEFT
        
        for step in range(5):
            # Update
            x1 += v1x * 0.5
            x2 += v2x * 0.5
            
            # Bounce
            if x1 < 3 or x1 > 29: v1x *= -1
            if x2 < 3 or x2 > 29: v2x *= -1
            
            x1 = max(3, min(29, x1))
            x2 = max(3, min(29, x2))
            
            # Image
            img = np.zeros((32, 32, 3), dtype=np.float32)
            img[int(y), int(x1)] = [1, 1, 1]  # Ball A (white)
            img[int(y), int(x2)] = [0.5, 0.5, 0.5]  # Ball B (gray)
            
            # Target: position of ball A (the one that started LEFT)
            target = [x1 / 32, y / 32]
            
            # Store step for later
            if step == 2:  # Middle step
                data.append((img.copy(), target.copy(), x1 < x2))  # x1 < x2 = A is left
    
    return data

# Create dataset
print("\n1. Creating random motion dataset...")
data = create_dataset(5000)
images = np.array([d[0] for d in data])
targets = np.array([d[1] for d in data])
# Check: after random motion, which ball is where?

X = torch.FloatTensor(images).permute(0, 3, 1, 2)
Y = torch.FloatTensor(targets)

print(f"   Dataset: {len(X)} samples")

# Models
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

class SlotFixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        
        # Fixed position embeddings
        self.slot_embed = nn.Parameter(torch.randn(2, 64) * 0.1)
        
        # FIXED binding
        self.pred_a = nn.Linear(64, 2)
        self.pred_b = nn.Linear(64, 2)
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slot_embed.unsqueeze(0)
        
        # FIXED: slot 0 -> ball A, slot 1 -> ball B
        pred_a = self.pred_a(h[:, 0, :])
        pred_b = self.pred_b(h[:, 1, :])
        
        return pred_a, pred_b

# Train
print("\n2. Training...")

results = {}

# Baseline
print("   Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(12):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p = m(X[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    p = m(X)
    results['Baseline'] = F.mse_loss(p, Y).item()

# SlotFixed
print("   SlotFixed...")
m = SlotFixed()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(12):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        pa, pb = m(X[idx[i:i+32]])
        
        # FIXED binding: pa should predict ball A
        loss = F.mse_loss(pa, Y[idx[i:i+32]])
        
        # Slot separation
        loss = loss + 0.3 * F.mse_loss(pa, pb)
        
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    pa, pb = m(X)
    results['SlotFixed'] = F.mse_loss(pa, Y).item()

# Summary
print("\n" + "="*60)
print("RESULTS")
print("="*60)
for name, mse in results.items():
    print(f"{name}: MSE = {mse:.4f}")

diff = results['Baseline'] - results['SlotFixed']
print(f"\nDifference: {diff:+.4f}")
if diff > 0.01:
    print("=> SlotFixed is BETTER!")
elif diff < -0.01:
    print("=> Baseline is BETTER!")
else:
    print("=> No significant difference")
