"""
CONTROL EXPERIMENT: Coordinates Input - Fixed
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
print("COORDINATES INPUT - FIXED")
print("="*60)

MAX_OBJ = 6

def make_features(positions, velocities):
    """Create padded feature vector: [obj0_x, obj0_y, obj0_vx, obj0_vy, ...]"""
    feat = []
    for i in range(MAX_OBJ):
        if i < len(positions):
            x, y = positions[i]
            vx, vy = velocities[i]
            feat.extend([x/32, y/32, vx/4, vy/4])
        else:
            feat.extend([0, 0, 0, 0])
    return feat

def generate_data(n, n_objects):
    X, Y = [], []
    for _ in range(n):
        positions = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        f0 = make_features(positions, velocities)
        
        for _ in range(10):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i], velocities[i] = (x, y), (vx, vy)
        
        f10 = make_features(positions, velocities)
        
        X.append(f0 + f10)  # 48 features
        Y.append(positions[0][0] / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("\n1. Generating data...")
X_train, Y_train = generate_data(2000, 2)
X_test_2, Y_test_2 = generate_data(1000, 2)
X_test_6, Y_test_6 = generate_data(1000, 6)

X_train = torch.FloatTensor(X_train)
X_test_2 = torch.FloatTensor(X_test_2)
X_test_6 = torch.FloatTensor(X_test_6)
Y_train = torch.FloatTensor(Y_train)

print(f"   Train: {X_train.shape}, Test2: {X_test_2.shape}, Test6: {X_test_6.shape}")

# Trajectory: flatten all
class TrajModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(x).squeeze()

# Object: process each object separately
class ObjModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16))
        self.out = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        B = x.size(0)
        # x: (B, 48) = t0(24) + t10(24)
        # Reshape: (B, 2, 24) -> (B, 2, 6, 4) -> (B, 6, 4) for each time
        x = x.view(B, 2, MAX_OBJ, 4)
        h = self.proc(x[:, 1])  # Use t10, shape (B, 6, 4)
        return self.out(h[:, 0]).squeeze()  # Object 0

print("\n2. Training TrajModel...")
m = TrajModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        p = m(X_train[idx[i:i+64]])
        loss = F.mse_loss(p, Y_train[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_t_2 = F.mse_loss(m(X_test_2), Y_test_2).item()
mse_t_6 = F.mse_loss(m(X_test_6), Y_test_6).item()

print("3. Training ObjModel...")
m = ObjModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        p = m(X_train[idx[i:i+64]])
        loss = F.mse_loss(p, Y_train[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_o_2 = F.mse_loss(m(X_test_2), Y_test_2).item()
mse_o_6 = F.mse_loss(m(X_test_6), Y_test_6).item()

r2, r6 = Y_test_2.var().item(), Y_test_6.var().item()

print("\n" + "="*60)
print("RESULTS (Coordinates - MSE)")
print("="*60)
print(f"Traj: 2obj={mse_t_2:.4f}, 6obj={mse_t_6:.4f}, drop={(mse_t_6-mse_t_2)/mse_t_2*100:.0f}%")
print(f"Obj:  2obj={mse_o_2:.4f}, 6obj={mse_o_6:.4f}, drop={(mse_o_6-mse_o_2)/mse_o_2*100:.0f}%")
