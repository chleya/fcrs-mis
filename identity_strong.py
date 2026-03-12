"""Identity Test - Stronger Supervision"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("="*60)
print("IDENTITY TEST - WITH SUPERVISION")
print("="*60)

# Simpler dataset: explicit tracking signal
print("\n1. Creating dataset with tracking signal...")

def create_dataset(n=4000):
    data = []
    for i in range(n):
        # Simple crossing scenario
        t = i * 0.02
        
        # Ball A starts left, Ball B starts right
        ball_a_x = 5.0 + t * 8  # Move right
        ball_b_x = 27.0 - t * 8  # Move left
        ball_y = 16.0
        
        # Add small vertical offset to make them distinguishable
        offset_a = np.sin(t) * 2
        offset_b = np.cos(t) * 2
        
        # Image
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[max(0,min(31,int(ball_y+offset_a))), max(0,min(31,int(ball_a_x)))] = [1,1,1]
        img[max(0,min(31,int(ball_y+offset_b))), max(0,min(31,int(ball_b_x)))] = [0.5,0.5,0.5]
        
        # Target: position of ball A (the one we track)
        target = [ball_a_x / 32, (ball_y + offset_a) / 32]
        
        # Track which ball is A (leftmost)
        is_a_left = 1 if ball_a_x < ball_b_x else 0
        
        data.append((img, target, is_a_left))
    
    return data

data = create_dataset(4000)
images = np.array([d[0] for d in data])
targets = np.array([d[1] for d in data])
track_signal = np.array([d[2] for d in data])

X = torch.FloatTensor(images).permute(0, 3, 1, 2)
Y = torch.FloatTensor(targets)

print(f"   Dataset: {len(X)} samples")

# Models
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        return self.fc(h), h

class SlotWithTracking(nn.Module):
    """Slot with explicit tracking loss"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        # Two slots
        self.slot_fc = nn.Linear(64*8*8, 64 * 2)
        
        # Each slot has its own predictor
        self.pred1 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.pred2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        
    def forward(self, x, return_slots=False):
        h = self.conv(x).reshape(x.size(0), -1)
        slots = self.slot_fc(h).reshape(x.size(0), 2, 64)
        
        # Predictions from each slot
        p1 = self.pred1(slots[:, 0, :])
        p2 = self.pred2(slots[:, 1, :])
        
        if return_slots:
            return p1, p2, slots
        return p1, p2

# Training
print("\n2. Training models...")

results = {}

# Baseline
print("\nBaseline:")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(12):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p, _ = m(X[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
m.eval()
with torch.no_grad():
    p, _ = m(X)
    results['Baseline'] = F.mse_loss(p, Y).item()
    print(f"   MSE: {results['Baseline']:.4f}")

# Slot with tracking
print("\nSlotWithTracking:")
m = SlotWithTracking()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(12):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p1, p2 = m(X[idx[i:i+32]])
        target = Y[idx[i:i+32]]
        
        # Main loss: predict ball A position
        # Model should learn which slot tracks ball A
        loss = F.mse_loss(p1, target) + F.mse_loss(p2, target)
        
        # Constraint: slots should be different
        slot_diff = F.mse_loss(p1, p2)
        loss = loss + 0.01 * slot_diff
        
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    p1, p2 = m(X)
    # Best prediction
    pred = p1  # Assume slot 1 tracks ball A
    results['Slot'] = F.mse_loss(pred, Y).item()
    print(f"   MSE: {results['Slot']:.4f}")
    
    # Check which slot is better
    mse1 = F.mse_loss(p1, Y).item()
    mse2 = F.mse_loss(p2, Y).item()
    print(f"   Slot1 MSE: {mse1:.4f}, Slot2 MSE: {mse2:.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, mse in results.items():
    print(f"{name}: MSE = {mse:.4f}")
