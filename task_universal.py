"""
TASK UNIVERSALITY: Different Physics Environments
Test if conclusions hold across different world types

Environment 1: Independent movers (low interaction)
Environment 2: Spring connections (medium interaction)  
Environment 3: Billiard physics (high interaction)
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
print("TASK UNIVERSALITY TEST")
print("="*60)

def clamp(v): return max(3, min(28, int(v)))

# Environment 1: Independent movers
def gen_independent(n, nobj):
    X, Y = [], []
    for _ in range(n):
        # Each ball moves independently
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(nobj)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(nobj)]
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img0[clamp(y), clamp(x)] = [1,1,1]
        
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]
                vx, vy = vel[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img10[clamp(y), clamp(x)] = [1,1,1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(pos[0][0]/32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Environment 2: Spring connections
def gen_spring(n, nobj):
    X, Y = [], []
    for _ in range(n):
        # Balls connected by springs
        pos = [(random.uniform(5+3*i, 10+3*i), random.uniform(10, 22)) for i in range(nobj)]
        vel = [(0, 0) for _ in range(nobj)]  # Start stationary
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img0[clamp(y), clamp(x)] = [1,1,1]
        
        # Spring physics
        for _ in range(10):
            for i in range(len(pos)):
                fx, fy = 0, 0
                # Spring force to neighbors
                for j in range(len(pos)):
                    if i != j:
                        dx = pos[j][0] - pos[i][0]
                        dy = pos[j][1] - pos[i][1]
                        dist = (dx**2 + dy**2)**0.5
                        if dist > 0.1:
                            fx += dx / dist * 0.1
                            fy += dy / dist * 0.1
                
                vx, vy = vel[i]
                vx += fx * 0.1
                vy += fy * 0.1
                vx *= 0.9  # Damping
                vy *= 0.9
                
                x, y = pos[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img10[clamp(y), clamp(x)] = [1,1,1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(pos[0][0]/32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Environment 3: Billiard (high interaction)
def gen_billiard(n, nobj):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(nobj)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(nobj)]
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img0[clamp(y), clamp(x)] = [1,1,1]
        
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]
                vx, vy = vel[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
            
            # Collision detection
            for i in range(len(pos)):
                for j in range(i+1, len(pos)):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = (dx**2 + dy**2)**0.5
                    if dist < 2:
                        # Elastic collision
                        vel[i], vel[j] = vel[j], vel[i]
        
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

print("\n1. Testing across environments...")

envs = [
    ("Independent", gen_independent),
    ("Spring", gen_spring),
    ("Billiard", gen_billiard)
]

results = []

for env_name, gen_func in envs:
    print(f"\n=== {env_name} ===")
    
    # Train on N=2
    X_tr, Y_tr = gen_func(1500, 2)
    X_tr = torch.FloatTensor(X_tr).permute(0,3,1,2)
    Y_tr = torch.FloatTensor(Y_tr)
    
    env_results = {'name': env_name, 'traj': [], 'obj': []}
    
    for n in [2, 4, 6, 8]:
        X_t, Y_t = gen_func(500, n)
        X_t = torch.FloatTensor(X_t).permute(0,3,1,2)
        Y_t = torch.FloatTensor(Y_t)
        
        # Trajectory
        m = TP()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(8):
            idx = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 64):
                p = m(X_tr[idx[i:i+64]])
                loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
                opt.zero_grad(); loss.backward(); opt.step()
        
        mse_t = F.mse_loss(m(X_t), Y_t).item()
        rnd = Y_t.var().item()
        pct_t = (rnd-mse_t)/rnd*100 if rnd>0 else 0
        
        # Object
        m = OP(n_slots=min(n+2,10))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(8):
            idx = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 64):
                p = m(X_tr[idx[i:i+64]])
                loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
                opt.zero_grad(); loss.backward(); opt.step()
        
        mse_o = F.mse_loss(m(X_t), Y_t).item()
        pct_o = (rnd-mse_o)/rnd*100 if rnd>0 else 0
        
        env_results['traj'].append(pct_t)
        env_results['obj'].append(pct_o)
        
        print(f"  N={n}: Traj={pct_t:+5.0f}%, Obj={pct_o:+5.0f}%")
    
    results.append(env_results)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for r in results:
    print(f"\n{r['name']}:")
    for i, n in enumerate([2, 4, 6, 8]):
        t, o = r['traj'][i], r['obj'][i]
        if o > t:
            print(f"  N={n}: Object wins by {o-t:.0f}%")
        else:
            print(f"  N={n}: Trajectory wins by {t-o:.0f}%")
