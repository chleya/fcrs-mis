"""
PERCEPTION BOTTLENECK - Simple Test
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42); np.random.seed(42); torch.manual_seed(42)

def clamp(v): return max(3, min(28, int(v)))

# Coordinates: given objects
def gen_coord(n, nobj):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(nobj)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(nobj)]
        
        f0 = []
        for i in range(6):
            if i < nobj: f0.extend([pos[i][0]/32, pos[i][1]/32])
            else: f0.extend([0, 0])
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]; vx, vy = vel[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
        
        f10 = []
        for i in range(6):
            if i < nobj: f10.extend([pos[i][0]/32, pos[i][1]/32])
            else: f10.extend([0, 0])
        
        X.append(f0 + f10)
        Y.append(pos[0][0]/32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Pixels
def gen_pix(n, nobj):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(nobj)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(nobj)]
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img0[clamp(y), clamp(x)] = [1,1,1]
        
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]; vx, vy = vel[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img10[clamp(y), clamp(x)] = [1,1,1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(pos[0][0]/32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

class MC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(24, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self,x): return self.fc(x).squeeze()

class MP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x): return self.fc(self.enc(x).flatten(1)).squeeze()

print("="*60)
print("PERCEPTION BOTTLENECK TEST")
print("="*60)

# Coordinates
print("\nCoordinates:")
Xc_tr, Yc = gen_coord(1500, 2)
Xc_tr = torch.FloatTensor(Xc_tr)
Yc = torch.FloatTensor(Yc)

for n in [2, 6]:
    Xc_t, Yc_t = gen_coord(500, n)
    Xc_t = torch.FloatTensor(Xc_t)
    Yc_t = torch.FloatTensor(Yc_t)
    
    m = MC()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(Xc_tr))
        for i in range(0, len(Xc_tr), 64):
            p = m(Xc_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Yc[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    pct = (Yc_t.var().item() - F.mse_loss(m(Xc_t), Yc_t).item()) / Yc_t.var().item() * 100
    print(f"  N={n}: {pct:+.0f}%")

# Pixels
print("\nPixels:")
Xp_tr, Yp = gen_pix(1500, 2)
Xp_tr = torch.FloatTensor(Xp_tr).permute(0,3,1,2)
Yp = torch.FloatTensor(Yp)

for n in [2, 6]:
    Xp_t, Yp_t = gen_pix(500, n)
    Xp_t = torch.FloatTensor(Xp_t).permute(0,3,1,2)
    Yp_t = torch.FloatTensor(Yp_t)
    
    m = MP()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(Xp_tr))
        for i in range(0, len(Xp_tr), 64):
            p = m(Xp_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Yp[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    pct = (Yp_t.var().item() - F.mse_loss(m(Xp_t), Yp_t).item()) / Yp_t.var().item() * 100
    print(f"  N={n}: {pct:+.0f}%")

print("\n=> Coordinates give object info directly")
print("=> Pixels require object discovery (perception bottleneck)")
