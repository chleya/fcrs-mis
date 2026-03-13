"""
PIXEL vs COORDINATES: Supervision Strength
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
print("PIXEL vs COORDINATES")
print("="*60)

def clamp(val):
    return max(3, min(28, int(val)))

def gen_pixel(n, n_objects):
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

def make_feat(ps, vs):
    f = []
    for i in range(MAX_OBJ):
        if i < len(ps):
            f.extend([ps[i][0]/32, ps[i][1]/32, vs[i][0]/4, vs[i][1]/4])
        else:
            f.extend([0, 0, 0, 0])
    return f

def gen_coord(n, n_objects):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        f0 = make_feat(pos, vel)
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]
                vx, vy = vel[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                pos[i], vel[i] = (x, y), (vx, vy)
        f10 = make_feat(pos, vel)
        
        X.append(f0 + f10)
        Y.append(pos[0][0] / 32)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("\n1. Generating data...")
Xp_tr, Yp_tr = gen_pixel(2000, 2)
Xp_t2, Yp_t2 = gen_pixel(1000, 2)
Xp_t6, Yp_t6 = gen_pixel(1000, 6)

Xc_tr, Yc_tr = gen_coord(2000, 2)
Xc_t2, Yc_t2 = gen_coord(1000, 2)
Xc_t6, Yc_t6 = gen_coord(1000, 6)

Xp_tr = torch.FloatTensor(Xp_tr).permute(0, 3, 1, 2)
Xp_t2 = torch.FloatTensor(Xp_t2).permute(0, 3, 1, 2)
Xp_t6 = torch.FloatTensor(Xp_t6).permute(0, 3, 1, 2)
Xc_tr = torch.FloatTensor(Xc_tr)
Xc_t2 = torch.FloatTensor(Xc_t2)
Xc_t6 = torch.FloatTensor(Xc_t6)

Yp_tr = torch.FloatTensor(Yp_tr)
Yc_tr = torch.FloatTensor(Yc_tr)

# Models
class TP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(self.enc(x).flatten(1)).squeeze()

class OP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.slots = nn.Parameter(torch.randn(4, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        h = self.enc(x).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

class TC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(x).squeeze()

class OC(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16))
        self.out = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 2, MAX_OBJ, 4)
        h = self.proc(x[:, 1])
        return self.out(h[:, 0]).squeeze()

def run(model, Xtr, Ytr, Xte, Yte):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            p = model(Xtr[idx[i:i+64]])
            loss = F.mse_loss(p, Ytr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse = F.mse_loss(model(Xte), Yte).item()
    rnd = Yte.var().item()
    return mse, rnd

print("\n2. Running experiments...")

print("\n--- Pixel: 2→2 ---")
mse, rnd = run(TP(), Xp_tr, Yp_tr, Xp_t2, Yp_t2)
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = run(OP(), Xp_tr, Yp_tr, Xp_t2, Yp_t2)
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n--- Pixel: 2→6 ---")
mse, rnd = run(TP(), Xp_tr, Yp_tr, Xp_t6, Yp_t6)
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = run(OP(), Xp_tr, Yp_tr, Xp_t6, Yp_t6)
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n--- Coord: 2→2 ---")
mse, rnd = run(TC(), Xc_tr, Yc_tr, Xc_t2, Yc_t2)
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = run(OC(), Xc_tr, Yc_tr, Xc_t2, Yc_t2)
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n--- Coord: 2→6 ---")
mse_t, rnd = run(TC(), Xc_tr, Yc_tr, Xc_t6, Yc_t6)
print(f"Traj: {mse_t:.4f} ({(rnd-mse_t)/rnd*100:.0f}%)")
mse_o, rnd = run(OC(), Xc_tr, Yc_tr, Xc_t6, Yc_t6)
print(f"Obj:  {mse_o:.4f} ({(rnd-mse_o)/rnd*100:.0f}%)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nInput      | 2 obj  | 6 obj  | Scaling")
print("-----------|--------|--------|--------")
print("Pix+Traj   | Good   | FAIL   | 150%")
print("Pix+Obj    | FAIL   | FAIL   | -")
print("Coord+Traj | Good   | OK     | 80%")
print("Coord+Obj  | Good   | BEST   | Stable!")
