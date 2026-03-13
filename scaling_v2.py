"""
LOG-SCALE ANALYSIS - Fixed
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("LOG-SCALE ANALYSIS")
print("="*60)

def clamp(v): return max(3, min(28, int(v)))

def generate_data(n, n_objects):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
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

print("\n1. Running experiments...")
X_tr, Y_tr = generate_data(2000, 2)
X_tr = torch.FloatTensor(X_tr).permute(0,3,1,2)
Y_tr = torch.FloatTensor(Y_tr)

ns = list(range(1, 11))
mse_t, mse_o, pct_t, pct_o = [], [], [], []

for n in ns:
    X_t, Y_t = generate_data(1000, n)
    X_t = torch.FloatTensor(X_t).permute(0,3,1,2)
    Y_t = torch.FloatTensor(Y_t)
    
    m = TP()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            p = m(X_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    mse_t.append(F.mse_loss(m(X_t), Y_t).item())
    
    m = OP(n_slots=min(n+2,10))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            p = m(X_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    mse_o.append(F.mse_loss(m(X_t), Y_t).item())
    
    rnd = Y_t.var().item()
    pct_t.append((rnd-mse_t[-1])/rnd*100 if rnd>0 else 0)
    pct_o.append((rnd-mse_o[-1])/rnd*100 if rnd>0 else 0)
    
    print(f"N={n}: Traj={mse_t[-1]:.4f}, Obj={mse_o[-1]:.4f}")

# Curve fitting
from scipy import stats

success_t = [max(1e-6, (rnd-mse_t[i])/rnd) for i, n in enumerate(ns)]
success_o = [max(1e-6, (rnd-mse_o[i])/rnd) for i, n in enumerate(ns)]

log_t = np.log(success_t)
log_o = np.log(success_o)

slope_t, intercept_t, r_t, p_t, se_t = stats.linregress(ns, log_t)
slope_o, intercept_o, r_o, p_o, se_o = stats.linregress(ns, log_o)

print(f"\n2. Curve Fitting:")
print(f"Trajectory: log(success) = {slope_t:.2f} * N + {intercept_t:.2f}, R2={r_t**2:.3f}")
print(f"Object:     log(success) = {slope_o:.2f} * N + {intercept_o:.2f}, R2={r_o**2:.3f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0,0]
ax1.plot(ns, pct_t, 'o-', label='Trajectory', color='#FF6B6B', linewidth=2)
ax1.plot(ns, pct_o, 's-', label='Object', color='#4ECDC4', linewidth=2)
ax1.axhline(y=0, color='black', linestyle='--')
ax1.set_xlabel('N (objects)')
ax1.set_ylabel('Performance (%)')
ax1.set_title('Linear Scale')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2 = axes[0,1]
ax2.semilogy(ns, mse_t, 'o-', label='Trajectory', color='#FF6B6B', linewidth=2)
ax2.semilogy(ns, mse_o, 's-', label='Object', color='#4ECDC4', linewidth=2)
ax2.set_xlabel('N (objects)')
ax2.set_ylabel('MSE (log)')
ax2.set_title('MSE vs N')
ax2.legend(); ax2.grid(True, alpha=0.3)

ax3 = axes[1,0]
ax3.semilogy(ns, success_t, 'o-', label='Trajectory', color='#FF6B6B', linewidth=2)
ax3.semilogy(ns, success_o, 's-', label='Object', color='#4ECDC4', linewidth=2)
ax3.set_xlabel('N (objects)')
ax3.set_ylabel('Success Rate (log)')
ax3.set_title('Success Rate')
ax3.legend(); ax3.grid(True, alpha=0.3)

ax4 = axes[1,1]
ax4.plot(ns, log_t, 'o-', label='Trajectory', color='#FF6B6B', linewidth=2)
ax4.plot(ns, log_o, 's-', label='Object', color='#4ECDC4', linewidth=2)
ax4.plot(ns, [intercept_t + slope_t*n for n in ns], '--', color='#FF6B6B', alpha=0.5)
ax4.plot(ns, [intercept_o + slope_o*n for n in ns], '--', color='#4ECDC4', alpha=0.5)
ax4.set_xlabel('N (objects)')
ax4.set_ylabel('log(Success)')
ax4.set_title(f'Log-Linear: Traj slope={slope_t:.2f}, Obj slope={slope_o:.2f}')
ax4.legend(); ax4.grid(True, alpha=0.3)

plt.suptitle('Scaling Law Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/fcrs_mis/scaling_law.png', dpi=150, bbox_inches='tight')
print("\nSaved: scaling_law.png")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
if abs(slope_t) > abs(slope_o) * 2:
    print(f"Trajectory decay rate: {abs(slope_t):.2f}")
    print(f"Object decay rate: {abs(slope_o):.2f}")
    print("=> TRAJECTORY HAS EXPONENTIAL SCALING FAILURE!")
    print(f"   Trajectory: success ~ exp({slope_t:.2f} * N)")
    print(f"   Object:     success ~ exp({slope_o:.2f} * N)")
