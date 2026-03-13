"""
MODEL CAPACITY vs TRANSITION SHARPNESS
Test if larger models have sharper phase transitions
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
print("MODEL CAPACITY vs TRANSITION SHARPNESS")
print("="*60)

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

# Different capacity models
class ModelSmall(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, hidden, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden*2, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(hidden*2*8*8, hidden*2), nn.ReLU(), nn.Linear(hidden*2, 1))
    def forward(self,x): return self.fc(self.enc(x).flatten(1)).squeeze()

class ModelMedium(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, hidden, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden*2, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(hidden*2*8*8, hidden*2), nn.ReLU(), nn.Linear(hidden*2, 1))
    def forward(self,x): return self.fc(self.enc(x).flatten(1)).squeeze()

class ModelLarge(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, hidden, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden*2, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(hidden*2*8*8, hidden*2), nn.ReLU(), nn.Linear(hidden*2, 1))
    def forward(self,x): return self.fc(self.enc(x).flatten(1)).squeeze()

print("\n1. Training on N=2, testing N=1-10...")

# Train data
X_tr, Y_tr = gen(2000, 2)
X_tr = torch.FloatTensor(X_tr).permute(0,3,1,2)
Y_tr = torch.FloatTensor(Y_tr)

results = []

for Model, name, hidden in [(ModelSmall, "Small", 32), (ModelMedium, "Medium", 64), (ModelLarge, "Large", 128)]:
    print(f"\n{name} Model (hidden={hidden}):")
    
    model_pcts = []
    
    for n in range(1, 11):
        X_t, Y_t = gen(500, n)
        X_t = torch.FloatTensor(X_t).permute(0,3,1,2)
        Y_t = torch.FloatTensor(Y_t)
        
        m = Model(hidden=hidden)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(10):
            idx = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 64):
                p = m(X_tr[idx[i:i+64]])
                loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
                opt.zero_grad(); loss.backward(); opt.step()
        
        mse = F.mse_loss(m(X_t), Y_t).item()
        rnd = Y_t.var().item()
        pct = (rnd-mse)/rnd*100 if rnd>0 else 0
        model_pcts.append(pct)
        
        print(f"  N={n}: {pct:+.0f}%")
    
    results.append({'name': name, 'pcts': model_pcts})

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

ns = list(range(1, 11))
colors = ['#3498DB', '#E74C3C', '#2ECC71']

for i, r in enumerate(results):
    ax.plot(ns, r['pcts'], 'o-', label=r['name'], color=colors[i], linewidth=2, markersize=8)

ax.axhline(y=0, color='black', linestyle='--')
ax.set_xlabel('N (objects)', fontsize=14)
ax.set_ylabel('Performance vs Random (%)', fontsize=14)
ax.set_title('Model Capacity vs Scaling Transition', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(ns)

plt.tight_layout()
plt.savefig('F:/fcrs_mis/capacity_test.png', dpi=150, bbox_inches='tight')
print("\nSaved: capacity_test.png")

# Analyze transition sharpness
print("\nTransition Analysis:")
for r in results:
    pcts = r['pcts']
    
    # Find where crosses zero
    cross = None
    for i in range(len(pcts)-1):
        if pcts[i] > 0 and pcts[i+1] <= 0:
            cross = i+1
            break
    
    # Calculate sharpness (drop rate)
    drops = []
    for i in range(1, len(pcts)):
        drop = pcts[i-1] - pcts[i]
        drops.append(drop)
    
    avg_drop = np.mean(drops) if drops else 0
    
    print(f"{r['name']}: crossover at N={cross}, avg drop={avg_drop:.1f}%")
