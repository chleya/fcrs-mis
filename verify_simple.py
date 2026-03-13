"""
VERIFY N_c vs 1/I - SIMPLIFIED
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42); np.random.seed(42); torch.manual_seed(42)

def clamp(v): return max(3, min(28, int(v)))

def gen(n, nobj, I=0.5):
    X, Y = [], []
    for _ in range(n):
        if I < 0.3:
            pos = [(random.uniform(5+3*i, 10+3*i), random.uniform(8, 24)) for i in range(nobj)]
        else:
            pos = [(random.uniform(5, 27), random.uniform(10+2*i, 12+2*i)) for i in range(nobj)]
        
        vel = [(random.uniform(-2, 2), random.uniform(-0.5, 0.5)) for _ in range(nobj)]
        
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]; vx, vy = vel[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
            
            if random.random() < I:
                for i in range(len(pos)):
                    for j in range(i+1, len(pos)):
                        d = ((pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)**0.5
                        if d < 3:
                            vel[i] = (vel[i][0]*-0.5, vel[i][1])
                            vel[j] = (vel[j][0]*-0.5, vel[j][1])
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img0[clamp(y), clamp(x)] = [1,1,1]
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img10[clamp(y), clamp(x)] = [1,1,1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(pos[0][0]/32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

class TP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x): return self.fc(self.enc(x).flatten(1)).squeeze()

class OP(nn.Module):
    def __init__(self, n_slots=6):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU())
        self.slots = nn.Parameter(torch.randn(n_slots,64)*0.1)
        self.predict = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x):
        h = self.enc(x).mean(dim=[2,3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:,0]).squeeze()

print("="*60)
print("VERIFY N_c vs 1/I")
print("="*60)

I_vals = [0.1, 0.3, 0.5, 0.8]
results = []

for I in I_vals:
    print(f"\nI={I}")
    
    X_tr, Y_tr = gen(1000, 2, I)
    X_tr = torch.FloatTensor(X_tr).permute(0,3,1,2)
    Y_tr = torch.FloatTensor(Y_tr)
    
    nc = None
    
    for n in [2, 4, 6, 8]:
        X_t, Y_t = gen(500, n, I)
        X_t = torch.FloatTensor(X_t).permute(0,3,1,2)
        Y_t = torch.FloatTensor(Y_t)
        
        # Traj
        m = TP()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(5):
            idx = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 64):
                p = m(X_tr[idx[i:i+64]])
                loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
                opt.zero_grad(); loss.backward(); opt.step()
        
        mse_t = F.mse_loss(m(X_t), Y_t).item()
        rnd = Y_t.var().item()
        pct_t = (rnd-mse_t)/rnd*100 if rnd>0 else 0
        
        # Obj
        m = OP(n_slots=min(n+2,10))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(5):
            idx = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 64):
                p = m(X_tr[idx[i:i+64]])
                loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
                opt.zero_grad(); loss.backward(); opt.step()
        
        mse_o = F.mse_loss(m(X_t), Y_t).item()
        pct_o = (rnd-mse_o)/rnd*100 if rnd>0 else 0
        
        print(f"  N={n}: T={pct_t:+.0f}%, O={pct_o:+.0f}%")
        
        if nc is None and pct_o > pct_t:
            nc = n
    
    results.append({'I': I, 'N_c': nc})
    print(f"  => N_c = {nc}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nI    N_c")
for r in results:
    print(f"{r['I']:.1f}  {r['N_c']}")
