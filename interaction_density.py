"""
INTERACTION DENSITY EXPERIMENT
Test how collision frequency affects scaling
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
print("INTERACTION DENSITY EXPERIMENT")
print("="*60)

def clamp(v): return max(3, min(28, int(v)))

def generate_data(n, n_objects, interaction_density=0.5):
    """
    interaction_density: 0 = no interaction, 1 = high interaction
    """
    X, Y = [], []
    for _ in range(n):
        # Initialize positions - spread out or clustered based on density
        if interaction_density < 0.3:
            # Low: spread out, rarely collide
            pos = [(random.uniform(5+3*i, 10+3*i), random.uniform(8, 24)) 
                   for i in range(n_objects)]
        else:
            # High: clustered, frequent collision
            pos = [(random.uniform(5, 27), random.uniform(10+2*i, 12+2*i)) 
                   for i in range(n_objects)]
        
        vel = [(random.uniform(-2, 2), random.uniform(-0.5, 0.5)) for _ in range(n_objects)]
        
        # Bounce balls off each other based on density
        for step in range(10):
            # Move
            for i in range(len(pos)):
                x, y = pos[i]
                vx, vy = vel[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                pos[i], vel[i] = (x, y), (vx, vy)
            
            # Simple collision based on density
            if random.random() < interaction_density:
                for i in range(len(pos)):
                    for j in range(i+1, len(pos)):
                        x1, y1 = pos[i]
                        x2, y2 = pos[j]
                        dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
                        if dist < 3:
                            # Collision!
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

print("\n1. Testing different interaction densities...")

densities = [0.0, 0.3, 0.6, 1.0]
n_objects = 6

results = []

for density in densities:
    print(f"\nDensity = {density}")
    
    # Generate data
    X_tr, Y_tr = generate_data(2000, 2, density)
    X_t, Y_t = generate_data(1000, n_objects, density)
    
    X_tr = torch.FloatTensor(X_tr).permute(0,3,1,2)
    X_t = torch.FloatTensor(X_t).permute(0,3,1,2)
    Y_tr = torch.FloatTensor(Y_tr)
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
    m = OP(n_slots=8)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            p = m(X_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    mse_o = F.mse_loss(m(X_t), Y_t).item()
    pct_o = (rnd-mse_o)/rnd*100 if rnd>0 else 0
    
    results.append({'density': density, 'traj': pct_t, 'obj': pct_o})
    print(f"  Traj: {pct_t:+.0f}%, Obj: {pct_o:+.0f}%")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nInteraction Density Effect on N=6 Scaling:")
print(f"{'Density':<10} {'Traj':<10} {'Obj':<10}")
for r in results:
    print(f"{r['density']:<10.1f} {r['traj']:+10.0f}% {r['obj']:+10.0f}%")

# Plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

dens = [r['density'] for r in results]
traj = [r['traj'] for r in results]
obj = [r['obj'] for r in results]

ax.plot(dens, traj, 'o-', label='Trajectory', color='#FF6B6B', linewidth=2, markersize=10)
ax.plot(dens, obj, 's-', label='Object', color='#4ECDC4', linewidth=2, markersize=10)

ax.axhline(y=0, color='black', linestyle='--')
ax.set_xlabel('Interaction Density', fontsize=14)
ax.set_ylabel('Performance vs Random (%)', fontsize=14)
ax.set_title('Effect of Interaction Density on Scaling (N=6)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('F:/fcrs_mis/interaction_density.png', dpi=150, bbox_inches='tight')
print("\nSaved: interaction_density.png")

# Analysis
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
for r in results:
    if r['obj'] > r['traj']:
        print(f"At density {r['density']}: Object wins by {r['obj']-r['traj']:.0f}%")
    else:
        print(f"At density {r['density']}: Trajectory wins by {r['traj']-r['obj']:.0f}%")
