"""
L4: INTERVENTION / COUNTERFACTUAL
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
print("L4: INTERVENTION")
print("="*60)

def clamp(v): return max(3, min(28, int(v)))

# Simple collision data
def gen_collision(n):
    X, Y = [], []
    for _ in range(n):
        xa, ya = random.uniform(5, 12), random.uniform(12, 20)
        xb, yb = random.uniform(16, 22), ya
        va = random.uniform(2, 4)
        vb = 0
        
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[clamp(ya), clamp(xa)] = [1, 0, 0]
        img0[clamp(yb), clamp(xb)] = [0, 0, 1]
        
        for _ in range(10):
            xa += va * 0.5
            xb += vb * 0.5
            if xa >= xb - 1 and va > 0:
                vb = va * 0.8
                va = -va * 0.3
                xa = xb - 1.1
        
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[clamp(ya), clamp(xa)] = [1, 1, 1]
        img10[clamp(yb), clamp(xb)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(xb / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def gen_counter(n):
    X, Y = [], []
    for _ in range(n):
        xa, ya = random.uniform(5, 12), random.uniform(12, 20)
        xb, yb = random.uniform(16, 22), ya
        va = random.uniform(2, 4)
        vb = 0
        
        # Counterfactual: B just moves straight
        xb_cf = xb
        for _ in range(10):
            xb_cf += vb * 0.5
            if xb_cf < 3 or xb_cf > 29: vb *= -1
        
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[clamp(ya), clamp(xa)] = [1, 0, 0]
        img0[clamp(yb), clamp(xb)] = [0, 0, 1]
        
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[clamp(yb), clamp(xb_cf)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(xb_cf / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("\n1. Generating data...")
Xf, Yf = gen_collision(2000)
Xc, Yc = gen_counter(2000)
Xf = torch.FloatTensor(Xf).permute(0, 3, 1, 2)
Xc = torch.FloatTensor(Xc).permute(0, 3, 1, 2)
Yf = torch.FloatTensor(Yf)
Yc = torch.FloatTensor(Yc)

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

def run(m, Xtr, Ytr, Xte, Yte):
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            p = m(Xtr[idx[i:i+64]])
            loss = F.mse_loss(p, Ytr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    pred = m(Xte)
    mse = ((pred - Yte) ** 2).mean().item()
    rnd = Yte.var().item()
    return mse, rnd

print("\n2. Training on factual, testing on counterfactual...")

print("\n--- Trajectory Model ---")
mse, rnd = run(TP(), Xf, Yf, Xc, Yc)
print(f"MSE: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n--- Object Model ---")
mse, rnd = run(OP(), Xf, Yf, Xc, Yc)
print(f"MSE: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n" + "="*60)
print("Counterfactual requires understanding object removal!")
