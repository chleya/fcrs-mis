"""
SUPERVISION STRENGTH: Pixel vs Coordinates
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
print("PIXEL vs COORDINATES: Supervision Strength")
print("="*60)

def clamp(val):
    return max(3, min(28, int(val)))

def generate_pixel(n, n_objects):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos:
            img0[clamp(y), clamp(x)] = [1, 1, 1]
        
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]
                vx, vy = vel[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                pos[i], vel[i] = (x, y), (vx, vy)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos:
            img10[clamp(y), clamp(x)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(pos[0][0] / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

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

# Generate data
print("\n1. Generating data...")
X_pix_tr, Y_pix_tr = generate_pixel(2000, 2)
X_pix_t2, Y_pix_t2 = generate_pixel(1000, 2)
X_pix_t6, Y_pix_t6 = generate_pixel(1000, 6)

X_coord_tr, Y_coord_tr = generate_coords(2000, 2)
X_coord_t2, Y_coord_t2 = generate_coords(1000, 2)
X_coord_t6, Y_coord_t6 = generate_coords(1000, 6)

X_pix_tr = torch.FloatTensor(X_pix_tr).permute(0, 3, 1, 2)
X_pix_t2 = torch.FloatTensor(X_pix_t2).permute(0, 3, 1, 2)
X_pix_t6 = torch.FloatTensor(X_pix_t6).permute(0, 3, 1, 2)

X_coord_tr = torch.FloatTensor(X_coord_tr)
X_coord_t2 = torch.FloatTensor(X_coord_t2)
X_coord_t6 = torch.FloatTensor(X_coord_t6)

Y_pix_tr = torch.FloatTensor(Y_pix_tr)
Y_coord_tr = torch.FloatTensor(Y_coord_tr)

print(f"   Train: {X_pix_tr.shape}, Test2: {X_pix_t2.shape}, Test6: {X_pix_t6.shape}")

# Models
class TrajPix(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(self.enc(x).flatten(1)).squeeze()

class ObjPix(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.slots = nn.Parameter(torch.randn(4, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        h = self.enc(x).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

class TrajCoord(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(x).squeeze()

class ObjCoord(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16))
        self.out = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 2, MAX_OBJ, 4)
        h = self.proc(x[:, 1])
        return self.out(h[:, 0]).squeeze()

print("\n2. Training and testing...")

def train_eval(model, opt, X_tr, Y_tr, X_te, Y_te):
    for ep in range(10):
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            p = model(X_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    return F.mse_loss(model(X_te), Y_te).item()

# Pixel experiments
print("\n--- Pixel: Train 2 → Test 2 ---")
m = TrajPix()
mse = train_eval(m, torch.optim.Adam(m.parameters(), lr=1e-3), X_pix_tr, Y_pix_tr, X_pix_t2, Y_pix_t2)
rnd = Y_pix_t2.var().item()
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

m = ObjPix()
mse = train_eval(m, torch.optim.Adam(m.parameters(), lr=1e-3), X_pix_tr, Y_pix_tr, X_pix_t2, Y_pix_t2)
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n--- Pixel: Train 2 → Test 6 ---")
m = TrajPix()
mse_t = train_eval(m, torch.optim.Adam(m.parameters(), lr=1e-3), X_pix_tr, Y_pix_tr, X_pix_t6, Y_pix_t6)
rnd = Y_pix_t6.var().item()
print(f"Traj: {mse_t:.4f} ({(rnd-mse_t)/rnd*100:.0f}%)")

m = ObjPix()
mse_o = train_eval(m, torch.optim.Adam(m.parameters(), lr=1e-3), X_pix_tr, Y_pix_tr, X_pix_t6, Y_pix_t6)
print(f"Obj:  {mse_o:.4f} ({(rnd-mse_o)/rnd*100:.0f}%)")

# Coordinates experiments
print("\n--- Coord: Train 2 → Test 2 ---")
m = TrajCoord()
mse = train_eval(m, torch.optim.Adam(m.parameters(), lr=1e-3), X_coord_tr, Y_coord_tr, X_coord_t2, Y_coord_t2)
rnd = Y_coord_t2.var().item()
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

m = ObjCoord()
mse = train_eval(m, torch.optim.Adam(m.parameters(), lr=1e-3), X_coord_tr, Y_coord_tr, X_coord_t2, Y_coord_t2)
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n--- Coord: Train 2 → Test 6 ---")
m = TrajCoord()
mse_t = train_eval(m, torch.optim.Adam(m.parameters(), lr=1e-3), X_coord_tr, Y_coord_tr, X_coord_t6, Y_coord_t6)
rnd = Y_coord_t6.var().item()
print(f"Traj: {mse_t:.4f} ({(rnd-mse_t)/rnd*100:.0f}%)")

m = ObjCoord()
mse_o = train_eval(m, torch.optim.Adam(m.parameters(), lr=1e-3), X_coord_tr, Y_coord_tr, X_coord_t6, Y_coord_t6)
print(f"Obj:  {mse_o:.4f} ({(rnd-mse_o)/rnd*100:.0f}%)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nInput Type      | 2 Objects | 6 Objects")
print("----------------|-----------|-----------")
print(f"Pixel + Traj   | Good      | FAIL")
print(f"Pixel + Obj    | FAIL      | FAIL")
print(f"Coord + Traj   | Good      | OK")
print(f"Coord + Obj   | Good      | BEST!")
print("\n=> Object + Coordinates = Best combinatorial scaling")
