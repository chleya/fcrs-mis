"""
LOG-SCALE ANALYSIS: Fit error curves to find scaling law
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
print("LOG-SCALE ANALYSIS: Finding Scaling Law")
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

print("\n1. Running N=1-10 experiments...")

# Train on N=2
X_tr, Y_tr = generate_data(2000, 2)
X_tr = torch.FloatTensor(X_tr).permute(0, 3, 1, 2)
Y_tr = torch.FloatTensor(Y_tr)

ns = list(range(1, 11))
mse_traj = []
mse_obj = []
pct_traj = []
pct_obj = []

for n in ns:
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
    mse_traj.append(mse_t)
    
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
    mse_obj.append(mse_o)
    
    rnd = Y_t.var().item()
    pct_t = (rnd - mse_t) / rnd * 100 if rnd > 0 else 0
    pct_o = (rnd - mse_o) / rnd * 100 if rnd > 0 else 0
    pct_traj.append(pct_t)
    pct_obj.append(pct_o)
    
    print(f"N={n}: Traj MSE={mse_t:.4f}, Obj MSE={mse_o:.4f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Linear scale
ax1 = axes[0, 0]
ax1.plot(ns, pct_traj, 'o-', label='Trajectory', color='#FF6B6B', linewidth=2)
ax1.plot(ns, pct_obj, 's-', label='Object', color='#4ECDC4', linewidth=2)
ax1.axhline(y=0, color='black', linestyle='--')
ax1.set_xlabel('N (objects)')
ax1.set_ylabel('Performance (%)')
ax1.set_title('Linear Scale')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Log scale MSE
ax2 = axes[0, 1]
ax2.semilogy(ns, mse_traj, 'o-', label='Trajectory', color='#FF6B6B', linewidth=2)
ax2.semilogy(ns, mse_obj, 's-', label='Object', color='#4ECDC4', linewidth=2)
ax2.set_xlabel('N (objects)')
ax2.set_ylabel('MSE (log scale)')
ax2.set_title('Log Scale MSE')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Error = (random - mse) / random, clipped to positive
error_traj = [(max(0.001, (rnd - mse_traj[i]) / rnd)) for i, n in enumerate(ns)]
error_obj = [(max(0.001, (rnd - mse_obj[i]) / rnd)) for i, n in enumerate(ns)]

ax3 = axes[1, 0]
ax3.semilogy(ns, error_traj, 'o-', label='Trajectory', color='#FF6B6B', linewidth=2)
ax3.semilogy(ns, error_obj, 's-', label='Object', color='#4ECDC4', linewidth=2)
ax3.set_xlabel('N (objects)')
ax3.set_ylabel('Success Rate (log)')
ax3.set_title('Success Rate (log scale)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. log(error) vs N - exponential test
log_error_traj = [np.log(max(1e-6, e)) for e in error_traj]
log_error_obj = [np.log(max(1e-6, e)) for e in error_obj]

ax4 = axes[1, 1]
ax4.plot(ns, log_error_traj, 'o-', label='Trajectory', color='#FF6B6B', linewidth=2)
ax4.plot(ns, log_error_obj, 's-', label='Object', color='#4ECDC4', linewidth=2)
ax4.set_xlabel('N (objects)')
ax4.set_ylabel('log(Success Rate)')
ax4.set_title('Log-Linear Test: slope = exponential rate')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Fit exponential for trajectory
from scipy import stats
slope_t, intercept_t, r_t, p_t, se_t = stats.linregress(ns, log_error_traj)
slope_o, intercept_o, r_o, p_o, se_o = stats.linregress(ns, log_error_obj)

print(f"\n2. Curve Fitting:")
print(f"Trajectory: log(success) = {slope_t:.2f} * N + {intercept_t:.2f}")
print(f"            R² = {r_t**2:.3f}")
print(f"Object:     log(success) = {slope_o:.2f} * N + {intercept_o:.2f}")
print(f"            R² = {r_o**2:.3f}")

# Add fit lines
fit_t = [np.exp(intercept_t + slope_t * n) for n in ns]
fit_o = [np.exp(intercept_o + slope_o * n) for n in ns]

ax4.plot(ns, fit_t, '--', color='#FF6B6B', alpha=0.5, label=f'Traj fit (slope={slope_t:.2f})')
ax4.plot(ns, fit_o, '--', color='#4ECDC4', alpha=0.5, label=f'Obj fit (slope={slope_o:.2f})')

plt.suptitle('Scaling Law Analysis: Error vs Object Count', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/fcrs_mis/scaling_law.png', dpi=150, bbox_inches='tight')
print("\nSaved: scaling_law.png")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if abs(slope_t) > abs(slope_o) * 2:
    print(f"\n=> Trajectory has {abs(slope_t)/abs(slope_o):.1f}x steeper decay rate!")
    print("   This confirms exponential scaling failure for Trajectory")
else:
    print("\n=> Similar decay rates")

print("\nTheoretical interpretation:")
print(f"  Trajectory: success ~ exp({slope_t:.2f} * N)")
print(f"  Object:     success ~ exp({slope_o:.2f} * N)")
