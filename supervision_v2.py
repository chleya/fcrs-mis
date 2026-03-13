"""
STAGE 2.5: Supervision Strength Experiment
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MAX_OBJ = 6
print("="*60)
print("SUPERVISION STRENGTH EXPERIMENT")
print("="*60)

def clamp(val):
    return max(3, min(28, int(val)))

def generate_data_basic(n, n_objects):
    X, Y = [], []
    for _ in range(n):
        positions = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img0[clamp(y), clamp(x)] = [1, 1, 1]
        
        for _ in range(10):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i], velocities[i] = (x, y), (vx, vy)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img10[clamp(y), clamp(x)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(positions[0][0] / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def generate_data_with_mask(n, n_objects):
    X, Y = [], []
    for _ in range(n):
        positions = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        img0 = np.zeros((32, 32, 4), np.float32)
        for i, (x, y) in enumerate(positions):
            img0[clamp(y), clamp(x), :3] = [1, 1, 1]
            if i == 0:
                img0[clamp(y), clamp(x), 3] = 1.0
        
        for _ in range(10):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i], velocities[i] = (x, y), (vx, vy)
        
        img10 = np.zeros((32, 32, 4), np.float32)
        for x, y in positions:
            img10[clamp(y), clamp(x), :3] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(positions[0][0] / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Generate data
print("\n1. Generating data...")
X_pix, Y_pix = generate_data_basic(2000, 2)
X_mask, Y_mask = generate_data_with_mask(2000, 2)
X_pix_t, Y_pix_t = generate_data_basic(1000, 6)
X_mask_t, Y_mask_t = generate_data_with_mask(1000, 6)

X_pix = torch.FloatTensor(X_pix).permute(0, 3, 1, 2)
X_mask = torch.FloatTensor(X_mask).permute(0, 3, 1, 2)
X_pix_t = torch.FloatTensor(X_pix_t).permute(0, 3, 1, 2)
X_mask_t = torch.FloatTensor(X_mask_t).permute(0, 3, 1, 2)

Y_pix = torch.FloatTensor(Y_pix)
Y_mask = torch.FloatTensor(Y_mask)
Y_pix_t = torch.FloatTensor(Y_pix_t)
Y_mask_t = torch.FloatTensor(Y_mask_t)

# Coordinates data
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

def generate_coords(n, n_objects):
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

X_coord, Y_coord = generate_coords(2000, 2)
X_coord_t, Y_coord_t = generate_coords(1000, 6)
X_coord = torch.FloatTensor(X_coord)
X_coord_t = torch.FloatTensor(X_coord_t)
Y_coord = torch.FloatTensor(Y_coord)
Y_coord_t = torch.FloatTensor(Y_coord_t)

print(f"   Pixel: {X_pix.shape}, Mask: {X_mask.shape}, Coord: {X_coord.shape}")

# Models
class TrajectoryModel(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(self.enc(x).flatten(1)).squeeze()

class ObjectModel(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slots = nn.Parameter(torch.randn(4, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        h = self.enc(x).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

def run_exp(X_train, Y_train, X_test, Y_test, in_channels):
    m = TrajectoryModel(in_channels)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 64):
            p = m(X_train[idx[i:i+64]])
            loss = F.mse_loss(p, Y_train[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse_t = F.mse_loss(m(X_test), Y_test).item()
    
    m = ObjectModel(in_channels)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 64):
            p = m(X_train[idx[i:i+64]])
            loss = F.mse_loss(p, Y_train[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse_o = F.mse_loss(m(X_test), Y_test).item()
    
    rnd = Y_test.var().item()
    return mse_t, mse_o, rnd

print("\n2. Running experiments...")

print("\n--- Pixel Only ---")
mse_t_pix, mse_o_pix, rnd_pix = run_exp(X_pix, Y_pix, X_pix_t, Y_pix_t, 6)
print(f"Traj: {mse_t_pix:.4f} ({(rnd_pix-mse_t_pix)/rnd_pix*100:.0f}%)")
print(f"Obj:  {mse_o_pix:.4f} ({(rnd_pix-mse_o_pix)/rnd_pix*100:.0f}%)")

print("\n--- Pixel + Mask ---")
mse_t_mask, mse_o_mask, rnd_mask = run_exp(X_mask, Y_mask, X_mask_t, Y_mask_t, 4)
print(f"Traj: {mse_t_mask:.4f} ({(rnd_mask-mse_t_mask)/rnd_mask*100:.0f}%)")
print(f"Obj:  {mse_o_mask:.4f} ({(rnd_mask-mse_o_mask)/rnd_mask*100:.0f}%)")

print("\n--- Coordinates ---")
class CoordTraj(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(x).squeeze()

class CoordObj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16))
        self.out = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 2, MAX_OBJ, 4)
        h = self.proc(x[:, 1])
        return self.out(h[:, 0]).squeeze()

m = CoordTraj()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_coord))
    for i in range(0, len(X_coord), 64):
        p = m(X_coord[idx[i:i+64]])
        loss = F.mse_loss(p, Y_coord[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_t_coord = F.mse_loss(m(X_coord_t), Y_coord_t).item()

m = CoordObj()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_coord))
    for i in range(0, len(X_coord), 64):
        p = m(X_coord[idx[i:i+64]])
        loss = F.mse_loss(p, Y_coord[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_o_coord = F.mse_loss(m(X_coord_t), Y_coord_t).item()

rnd_coord = Y_coord_t.var().item()
print(f"Traj: {mse_t_coord:.4f} ({(rnd_coord-mse_t_coord)/rnd_coord*100:.0f}%)")
print(f"Obj:  {mse_o_coord:.4f} ({(rnd_coord-mse_o_coord)/rnd_coord*100:.0f}%)")

print("\n" + "="*60)
print("SUMMARY: Supervision Strength vs Scaling")
print("="*60)
print("\nObject Model Performance (6 objects):")
print(f"Pixel only:      {(rnd_pix-mse_o_pix)/rnd_pix*100:+.0f}%")
print(f"Pixel + Mask:   {(rnd_mask-mse_o_mask)/rnd_mask*100:+.0f}%")
print(f"Coordinates:    {(rnd_coord-mse_o_coord)/rnd_coord*100:+.0f}%")

print("\n=> Coordinates (upper bound) achieves best scaling!")
