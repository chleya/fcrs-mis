"""Identity Test - HARD with REAL crossing"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("="*60)
print("IDENTITY TEST - REAL CROSSING")
print("="*60)

# Dataset with REAL crossing (positions actually swap)
def create_dataset(n=6000):
    data = []
    
    for i in range(n):
        t = i * 0.1  # Time
        
        # Ball A starts left, moves right then left
        # Ball B starts right, moves left then right
        # They CROSS and reverse
        
        if t < 5:  # Before crossing
            ball_a_x = 5.0 + t * 2
            ball_b_x = 27.0 - t * 2
        else:  # After crossing - POSITIONS SWAPPED!
            t2 = t - 5
            ball_a_x = 17.0 - t2 * 2  # Now going left
            ball_b_x = 15.0 + t2 * 2  # Now going right
        
        ball_y = 16.0
        
        # Clamp
        ball_a_x = max(2, min(30, ball_a_x))
        ball_b_x = max(2, min(30, ball_b_x))
        
        # Image
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[int(ball_y), int(ball_a_x)] = [1.0, 1.0, 1.0]  # Ball A (white)
        img[int(ball_y), int(ball_b_x)] = [0.5, 0.5, 0.5]  # Ball B (gray)
        
        # CRITICAL: After crossing, ball A is now on the RIGHT
        # If model doesn't track identity, it will predict LEFT
        target = [ball_a_x / 32, ball_y / 32]
        
        # Track which is harder
        crossing = 1 if t >= 5 else 0
        
        data.append((img, target, crossing))
    
    return data

# Create dataset
print("\n1. Creating dataset with real crossing...")
data = create_dataset(6000)
images = np.array([d[0] for d in data])
targets = np.array([d[1] for d in data])
crossing = np.array([d[2] for d in data])

X = torch.FloatTensor(images).permute(0, 3, 1, 2)
Y = torch.FloatTensor(targets)

print(f"   Total: {len(X)}, Crossing: {crossing.sum()}, Normal: {(crossing==0).sum()}")

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
        
        # FIXED position embeddings - key innovation!
        self.slot_embed = nn.Parameter(torch.randn(2, 64) * 0.1)
        
        # FIXED binding: slot 0 -> ball A, slot 1 -> ball B
        self.pred_a = nn.Linear(64, 2)  # Always predict ball A
        self.pred_b = nn.Linear(64, 2)  # Always predict ball B
        
    def forward(self, x):
        h = self.encoder(x)  # (B, 64, 8, 8)
        h = h.mean(dim=[2, 3])  # (B, 64)
        
        # Add fixed position embedding
        h = h.unsqueeze(1) + self.slot_embed.unsqueeze(0)  # (B, 2, 64)
        
        # FIXED prediction: slot 0 = ball A, slot 1 = ball B
        pred_a = self.pred_a(h[:, 0, :])  # Always ball A
        pred_b = self.pred_b(h[:, 1, :])  # Always ball B
        
        return pred_a, pred_b

# Training
print("\n2. Training...")

results = {}

# Baseline
print("   Training Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(10):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p = m(X[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    p = m(X)
    results['Baseline'] = {
        'all': F.mse_loss(p, Y).item(),
        'cross': F.mse_loss(p[crossing==1], Y[crossing==1]).item(),
        'normal': F.mse_loss(p[crossing==0], Y[crossing==0]).item()
    }

# SlotFixed
print("   Training SlotFixed...")
m = SlotFixed()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(10):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        pa, pb = m(X[idx[i:i+32]])
        
        # Main loss: predict ball A with slot 0 (FIXED binding!)
        loss = F.mse_loss(pa, Y[idx[i:i+32]])
        
        # Constraint: make slots different
        sep = F.mse_loss(pa, pb)
        loss = loss + 0.3 * sep
        
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    pa, pb = m(X)
    # Use slot 0 which should track ball A
    results['SlotFixed'] = {
        'all': F.mse_loss(pa, Y).item(),
        'cross': F.mse_loss(pa[crossing==1], Y[crossing==1]).item(),
        'normal': F.mse_loss(pa[crossing==0], Y[crossing==0]).item()
    }

# Summary
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"{'Model':<15} | {'All':>10} | {'Cross':>10} | {'Normal':>10}")
print("-"*55)
for name, r in results.items():
    print(f"{name:<15} | {r['all']:>10.4f} | {r['cross']:>10.4f} | {r['normal']:>10.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("Cross = positions actually swapped (hard!)")
print("Normal = before crossing")
print("\nIf SlotFixed << Baseline on Cross:")
print("  -> Fixed position embedding enables identity tracking!")
