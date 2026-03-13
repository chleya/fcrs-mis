"""
L5: CAUSAL TRAINING
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
print("L5: CAUSAL TRAINING")
print("="*60)

def clamp(v): return max(3, min(28, int(v)))

# Prediction data
def gen_pred(n):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(2)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(2)]
        img0 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img0[clamp(y), clamp(x)] = [1,1,1]
        for _ in range(10):
            for i in range(2):
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

# Intervention: remove one object
def gen_int(n):
    X, Y = [], []
    for _ in range(n):
        xa, ya = random.uniform(5, 12), random.uniform(12, 20)
        xb, yb = random.uniform(16, 22), ya
        va = random.uniform(2, 4)
        
        # With A
        xb_w = xb
        for _ in range(10):
            xa += va*0.5
            xb_w += 0
            if xa >= xb_w-1 and va>0:
                va = -va*0.3
                xa = xb_w-1.1
        
        # Without A (intervention)
        xb_wo = xb
        for _ in range(10):
            xb_wo += 0
        
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[clamp(ya), clamp(xa)] = [1,0,0]
        img0[clamp(yb), clamp(xb)] = [0,0,1]
        
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[clamp(ya), clamp(xa)] = [1,1,1]
        img10[clamp(yb), clamp(xb_w)] = [1,1,1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(xb_w/32)
        
        # Intervention input
        img0i = np.zeros((32, 32, 3), np.float32)
        img0i[clamp(yb), clamp(xb)] = [0,0,1]
        
        img10i = np.zeros((32, 32, 3), np.float32)
        img10i[clamp(yb), clamp(xb_wo)] = [1,1,1]
        
        X.append(np.concatenate([img0i, img10i], axis=2))
        Y.append(xb_wo/32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("\n1. Generating data...")
Xp, Yp = gen_pred(2000)
Xi, Yi = gen_int(1000)

Xp = torch.FloatTensor(Xp).permute(0,3,1,2)
Xi = torch.FloatTensor(Xi).permute(0,3,1,2)
Yp = torch.FloatTensor(Yp)
Yi = torch.FloatTensor(Yi)

class TP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x): return self.fc(self.enc(x).flatten(1)).squeeze()

class OP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU())
        self.slots = nn.Parameter(torch.randn(4,64)*0.1)
        self.predict = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x):
        h = self.enc(x).mean(dim=[2,3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:,0]).squeeze()

def run(model, Xtr, Ytr, Xte, Yte):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            p = model(Xtr[idx[i:i+64]])
            loss = F.mse_loss(p, Ytr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        mse = F.mse_loss(model(Xte), Yte).item()
    rnd = Yte.var().item()
    return mse, rnd

print("\n2. Prediction only → Test intervention...")
mse, rnd = run(TP(), Xp, Yp, Xi, Yi)
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = run(OP(), Xp, Yp, Xi, Yi)
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n3. Causal training (pred + int) → Test intervention...")
# Mix prediction and intervention for training
Xmix = torch.cat([Xp[:1000], Xi[:1000]], 0)
Ymix = torch.cat([Yp[:1000], Yi[:1000]], 0)

mse, rnd = run(TP(), Xmix, Ymix, Xi, Yi)
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = run(OP(), Xmix, Ymix, Xi, Yi)
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n" + "="*60)
print("Does causal training improve intervention reasoning?")
