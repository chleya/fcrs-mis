"""
CRITICAL POINT EXPERIMENT: N = 1 to 10
Find the critical N_c where Trajectory representation fails
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MAX_OBJ = 10
print("="*60)
print("CRITICAL POINT TEST: N = 1 to 10")
print("="*60)

def clamp(v): return max(3, min(28, int(v)))

def generate_data(n, n_objects):
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

# Models
class TrajModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(self.enc(x).flatten(1)).squeeze()

class ObjModel(nn.Module):
    def __init__(self, n_slots=6):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.slots = nn.Parameter(torch.randn(n_slots, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        h = self.enc(x).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

print("\n1. Training on N=2, testing on N=1-10...")

# Train on N=2
X_tr, Y_tr = generate_data(2000, 2)
X_tr = torch.FloatTensor(X_tr).permute(0, 3, 1, 2)
Y_tr = torch.FloatTensor(Y_tr)

results = []

for n in range(1, 11):
    X_t, Y_t = generate_data(1000, n)
    X_t = torch.FloatTensor(X_t).permute(0, 3, 1, 2)
    Y_t = torch.FloatTensor(Y_t)
    
    # Trajectory
    m = TrajModel()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            p = m(X_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    mse_t = F.mse_loss(m(X_t), Y_t).item()
    rnd_t = Y_t.var().item()
    pct_t = (rnd_t - mse_t) / rnd_t * 100 if rnd_t > 0 else 0
    
    # Object
    m = ObjModel(n_slots=min(n+2, 10))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            p = m(X_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    mse_o = F.mse_loss(m(X_t), Y_t).item()
    rnd_o = Y_t.var().item()
    pct_o = (rnd_o - mse_o) / rnd_o * 100 if rnd_o > 0 else 0
    
    results.append({
        'n': n,
        'traj': pct_t,
        'obj': pct_o,
        'diff': pct_o - pct_t
    })
    
    print(f"N={n:2d}: Traj={pct_t:+5.0f}%, Obj={pct_o:+5.0f}%, Diff={pct_o-pct_t:+5.0f}%")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Find critical point
for r in results:
    if r['traj'] < 0 and r['obj'] > 0:
        print(f"Critical point: N_c = {r['n']}")
        print(f"  At N={r['n']}: Trajectory fails, Object succeeds")
        break
else:
    print("No clear critical point found")

# Plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

ns = [r['n'] for r in results]
traj = [r['traj'] for r in results]
obj = [r['obj'] for r in results]

ax.plot(ns, traj, 'o-', label='Trajectory', color='#FF6B6B', linewidth=2, markersize=8)
ax.plot(ns, obj, 's-', label='Object', color='#4ECDC4', linewidth=2, markersize=8)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.fill_between(ns, traj, obj, alpha=0.2, color='green', label='Object advantage')

ax.set_xlabel('Number of Objects (N)', fontsize=14)
ax.set_ylabel('Performance vs Random (%)', fontsize=14)
ax.set_title('Critical Point Analysis: Representation Phase Transition', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.set_xticks(ns)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('F:/fcrs_mis/critical_point.png', dpi=150, bbox_inches='tight')
print("\nSaved: critical_point.png")

# Fit curves
print("\nCurve fitting:")
traj_arr = np.array(traj)
ns_arr = np.array(ns)

# For trajectory: find where it crosses zero
for i in range(len(traj_arr) - 1):
    if traj_arr[i] > 0 and traj_arr[i+1] <= 0:
        # Linear interpolation
        n_c = ns[i] + (0 - traj_arr[i]) / (traj_arr[i+1] - traj_arr[i])
        print(f"Trajectory critical N_c ≈ {n_c:.1f}")
