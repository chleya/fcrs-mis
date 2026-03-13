"""
OBJECT MODEL COMPARISON ON COMBINATORIAL SCALING
Key: Does object-factored model generalize better than trajectory model when object count changes?
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
print("OBJECT MODEL vs TRAJECTORY ON SCALING")
print("="*60)

def clamp(val):
    return max(3, min(28, int(val)))

def generate_data(n, n_objects):
    X, Y = [], []
    for _ in range(n):
        positions = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for i, (x, y) in enumerate(positions):
            if i == 0:
                img0[clamp(y), clamp(x)] = [1, 0, 0]
            else:
                img0[clamp(y), clamp(x)] = [0, 0, 1]
        
        for step in range(10):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx * 0.5, y + vy * 0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i] = (x, y)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img10[clamp(y), clamp(x)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(positions[0][0] / 32)
    
    return np.array(X), np.array(Y)

print("\n1. Generating data...")
X_train, Y_train = generate_data(2000, 2)
X_test_2, Y_test_2 = generate_data(1000, 2)
X_test_6, Y_test_6 = generate_data(1000, 6)

X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
X_test_2 = torch.FloatTensor(X_test_2).permute(0, 3, 1, 2)
X_test_6 = torch.FloatTensor(X_test_6).permute(0, 3, 1, 2)
Y_train = torch.FloatTensor(Y_train)
Y_test_2 = torch.FloatTensor(Y_test_2)
Y_test_6 = torch.FloatTensor(Y_test_6)

print(f"   Train (2 obj): {X_train.shape}")
print(f"   Test (2 obj): {X_test_2.shape}")
print(f"   Test (6 obj): {X_test_6.shape}")

# Model 1: Trajectory (Baseline)
class TrajectoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 1))
    
    def forward(self, x):
        return self.fc(self.enc(x).flatten(1)).squeeze()

# Model 2: Object-Factored (Slot)
class ObjectModel(nn.Module):
    def __init__(self, n_slots=6):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slots = nn.Parameter(torch.randn(n_slots, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x):
        h = self.enc(x).mean(dim=[2, 3])  # (B, 64)
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)  # (B, n_slots, 64)
        # Predict from all slots, use slot 0
        preds = self.predict(h)  # (B, n_slots, 1)
        return preds[:, 0].squeeze()

print("\n2. Training Trajectory Model...")
m = TrajectoryModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        p = m(X_train[idx[i:i+64]])
        loss = F.mse_loss(p, Y_train[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    mse_traj_2 = F.mse_loss(m(X_test_2), Y_test_2).item()
    mse_traj_6 = F.mse_loss(m(X_test_6), Y_test_6).item()

print("3. Training Object Model...")
m = ObjectModel(n_slots=6)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        p = m(X_train[idx[i:i+64]])
        loss = F.mse_loss(p, Y_train[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    mse_obj_2 = F.mse_loss(m(X_test_2), Y_test_2).item()
    mse_obj_6 = F.mse_loss(m(X_test_6), Y_test_6).item()

random_2 = Y_test_2.var().item()
random_6 = Y_test_6.var().item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nTrajectory Model:")
print(f"  Test (2 obj): MSE = {mse_traj_2:.4f} ({(random_2-mse_traj_2)/random_2*100:.1f}%)")
print(f"  Test (6 obj): MSE = {mse_traj_6:.4f} ({(random_6-mse_traj_6)/random_6*100:.1f}%)")
print(f"  Drop: {(mse_traj_6-mse_traj_2)/mse_traj_2*100:.1f}%")

print(f"\nObject Model:")
print(f"  Test (2 obj): MSE = {mse_obj_2:.4f} ({(random_2-mse_obj_2)/random_2*100:.1f}%)")
print(f"  Test (6 obj): MSE = {mse_obj_6:.4f} ({(random_6-mse_obj_6)/random_6*100:.1f}%)")
print(f"  Drop: {(mse_obj_6-mse_obj_2)/mse_obj_2*100:.1f}%")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse_obj_6 < mse_traj_6:
    print("=> Object model BETTER on 6 objects!")
    print("=> Object representation enables combinatorial generalization")
else:
    print("=> No significant advantage from object model")
    print("=> Both models struggle with scaling")
