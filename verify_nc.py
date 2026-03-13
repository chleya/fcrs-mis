"""
VERIFY N_c vs 1/I RELATIONSHIP
Find crossover point at different interaction densities
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
print("VERIFY N_c vs 1/I RELATIONSHIP")
print("="*60)

def clamp(v): return max(3, min(28, int(v)))

def generate_data(n, n_objects, interaction=0.5):
    X, Y = [], []
    for _ in range(n):
        if interaction < 0.3:
            pos = [(random.uniform(5+3*i, 10+3*i), random.uniform(8, 24)) for i in range(n_objects)]
        else:
            pos = [(random.uniform(5, 27), random.uniform(10+2*i, 12+2*i)) for i in range(n_objects)]
        
        vel = [(random.uniform(-2, 2), random.uniform(-0.5, 0.5)) for _ in range(n_objects)]
        
        for step in range(10):
            for i in range(len(pos)):
                x, y = pos[i]; vx, vy = vel[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
            
            if random.random() < interaction:
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

print("\n1. Finding crossover N_c for different I values...")

I_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
results = []

for I in I_values:
    print(f"\nI = {I}")
    
    # Train on N=2
    X_tr, Y_tr = generate_data(2000, 2, I)
    X_tr = torch.FloatTensor(X_tr).permute(0,3,1,2)
    Y_tr = torch.FloatTensor(Y_tr)
    
    crossover_found = None
    
    for n in range(1, 11):
        X_t, Y_t = generate_data(500, n, I)
        X_t = torch.FloatTensor(X_t).permute(0,3,1,2)
        Y_t = torch.FloatTensor(Y_t)
        
        # Trajectory
        m = TP()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(10):
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
        for ep in range(10):
            idx = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 64):
                p = m(X_tr[idx[i:i+64]])
                loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
                opt.zero_grad(); loss.backward(); opt.step()
        
        mse_o = F.mse_loss(m(X_t), Y_t).item()
        pct_o = (rnd-mse_o)/rnd*100 if rnd>0 else 0
        
        # Find crossover
        if crossover_found is None:
            if pct_o > pct_t:
                crossover_found = n
        
        if n <= 4 or crossover_found:
            print(f"  N={n}: Traj={pct_t:+5.0f}%, Obj={pct_o:+5.0f}%")
    
    results.append({'I': I, 'N_c': crossover_found})
    print(f"  => N_c = {crossover_found}")

# Plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Extract data
Is = [r['I'] for r in results if r['N_c'] is not None]
N_cs = [r['N_c'] for r in results if r['N_c'] is not None]

# Plot N_c vs I
ax1 = axes[0]
ax1.plot(Is, N_cs, 'o-', color='#9B59B6', linewidth=2, markersize=10)
ax1.set_xlabel('Interaction Density (I)', fontsize=14)
ax1.set_ylabel('Critical N_c', fontsize=14)
ax1.set_title('N_c vs I', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot N_c vs 1/I
ax2 = axes[1]
inv_I = [1/I for I in Is]
ax2.plot(inv_I, N_cs, 'o-', color='#E74C3C', linewidth=2, markersize=10)
ax2.set_xlabel('1/I', fontsize=14)
ax2.set_ylabel('Critical N_c', fontsize=14)
ax2.set_title('N_c vs 1/I (Test Linear Relationship)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Fit linear
from scipy import stats
slope, intercept, r, p, se = stats.linregress(inv_I, N_cs)
x_line = np.linspace(min(inv_I), max(inv_I), 10)
y_line = [slope*x + intercept for x in x_line]
ax2.plot(x_line, y_line, '--', color='gray', alpha=0.5, label=f'Linear fit (R2={r**2:.2f})')
ax2.legend()

plt.suptitle('Verifying N_c proportional to 1/I', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/fcrs_mis/verify_Nc.png', dpi=150, bbox_inches='tight')
print("\nSaved: verify_Nc.png")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
print(f"\nLinear fit: N_c = {slope:.2f} * (1/I) + {intercept:.2f}")
print(f"R-squared: {r**2:.3f}")

if r**2 > 0.7:
    print("\n=> STRONG evidence for N_c proportional to 1/I!")
    print("   Your theoretical model is VERIFIED!")
else:
    print("\n=> Weak correlation, need more data points")
