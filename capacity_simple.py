"""
CAPACITY TEST - Simple
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
random.seed(42); np.random.seed(42); torch.manual_seed(42)

def clamp(v): return max(3, min(28, int(v)))

def gen(n, nobj):
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

class Model(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6,h,3,2,1), nn.ReLU(), nn.Conv2d(h,h*2,3,2,1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(h*2*8*8,h*2), nn.ReLU(), nn.Linear(h*2,1))
    def forward(self,x): return self.fc(self.enc(x).flatten(1)).squeeze()

print("="*60)
print("CAPACITY TEST")
print("="*60)

X_tr, Y_tr = gen(1500, 2)
X_tr = torch.FloatTensor(X_tr).permute(0,3,1,2)
Y_tr = torch.FloatTensor(Y_tr)

for h, name in [(32,"Small"), (64,"Medium"), (128,"Large")]:
    print(f"\n{name} (h={h}):")
    pcts = []
    for n in [2, 4, 6, 8]:
        X_t, Y_t = gen(400, n)
        X_t = torch.FloatTensor(X_t).permute(0,3,1,2)
        Y_t = torch.FloatTensor(Y_t)
        m = Model(h)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(5):
            idx = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 64):
                p = m(X_tr[idx[i:i+64]])
                loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
                opt.zero_grad(); loss.backward(); opt.step()
        mse = F.mse_loss(m(X_t), Y_t).item()
        pct = (Y_t.var().item()-mse)/Y_t.var().item()*100
        pcts.append(pct)
        print(f"  N={n}: {pct:+.0f}%")
