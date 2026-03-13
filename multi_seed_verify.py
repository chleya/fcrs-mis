"""
MULTI-SEED STABILITY VERIFICATION
Test: 2→6 scaling with 5 seeds
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

MAX_OBJ = 6

def make_features(positions, velocities):
    feat = []
    for i in range(MAX_OBJ):
        if i < len(positions):
            x, y = positions[i]
            vx, vy = velocities[i]
            feat.extend([x/32, y/32, vx/4, vy/4])
        else:
            feat.extend([0, 0, 0, 0])
    return feat

def generate_data(n, n_objects, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        f0 = make_features(pos, vel)
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]
                vx, vy = vel[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                pos[i], vel[i] = (x, y), (vx, vy)
        f10 = make_features(pos, vel)
        X.append(f0 + f10)
        Y.append(pos[0][0] / 32)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

class TrajModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(x).squeeze()

class ObjModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16))
        self.out = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 2, MAX_OBJ, 4)
        h = self.proc(x[:, 1])
        return self.out(h[:, 0]).squeeze()

print("="*60)
print("MULTI-SEED VERIFICATION (5 seeds)")
print("="*60)

seeds = [42, 123, 456, 789, 1000]
traj_6_results = []
obj_6_results = []

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X_tr, Y_tr = generate_data(2000, 2, seed)
    X_t6, Y_t6 = generate_data(1000, 6, seed)
    
    X_tr = torch.FloatTensor(X_tr)
    X_t6 = torch.FloatTensor(X_t6)
    Y_tr = torch.FloatTensor(Y_tr)
    Y_t6 = torch.FloatTensor(Y_t6)
    
    # Traj
    m = TrajModel()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            p = m(X_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse_t = F.mse_loss(m(X_t6), Y_t6).item()
    traj_6_results.append(mse_t)
    
    # Obj
    m = ObjModel()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            p = m(X_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse_o = F.mse_loss(m(X_t6), Y_t6).item()
    obj_6_results.append(mse_o)
    
    print(f"Seed {seed}: Traj={mse_t:.4f}, Obj={mse_o:.4f}")

print("\n" + "="*60)
print("SUMMARY (6 objects, 5 seeds)")
print("="*60)
traj_mean, traj_std = np.mean(traj_6_results), np.std(traj_6_results)
obj_mean, obj_std = np.mean(obj_6_results), np.std(obj_6_results)

print(f"Trajectory: MSE = {traj_mean:.4f} ± {traj_std:.4f}")
print(f"Object:     MSE = {obj_mean:.4f} ± {obj_std:.4f}")
print(f"\nRatio: Object/Traj = {obj_mean/traj_mean:.2f}x better")

if obj_mean < traj_mean * 0.5:
    print("\n=> Object model SIGNIFICANTLY better (stable across seeds)!")
else:
    print("\n=> Difference less significant")
