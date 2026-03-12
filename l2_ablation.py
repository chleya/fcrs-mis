"""
L2 Root Cause Diagnosis - Ablation Study
Systematically test which components affect mechanism formation

Components to ablate:
1. Composition Loss weight (0, 0.1, 0.2, 0.3, 0.5)
2. Dynamics architecture (additive vs multiplicative)
3. Intervention strength (soft vs hard)
4. Latent dimension (2, 4, 8)
5. Training epochs (5, 10, 15)
6. Learning rate (1e-4, 3e-4, 1e-3)

Run: python l2_ablation.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print('='*60)
print('L2 ABLATION STUDY')
print('='*60)

# CartPole
def step(s, a):
    x, xd, t, td = s
    f = 10.0 if a==1 else -10.0
    ct, st = np.cos(t), np.sin(t)
    tmp = f/1.1 + 0.05*td**2*st
    ta = (9.8*st - ct*tmp)/(0.5*(4/3-0.1*ct**2/1.1))
    xa = tmp - 0.05*ta*ct/1.1
    return np.array([x+0.02*xd, xd+0.02*xa, t+0.02*td, td+0.02*ta])

# Generate data
print('\nGenerating data...')
data = []
for _ in range(2000):
    s = np.random.uniform(-0.05, 0.05, 4)
    for _ in range(20):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        data.append((s.copy(), a, sn.copy(), 'normal'))
        s = sn

# Add interventions
for _ in range(500):
    s = np.random.uniform(-0.05, 0.05, 4)
    s[2] = 0.0
    for _ in range(10):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        sn[2] = 0.0
        data.append((s.copy(), a, sn.copy(), 'theta_fix'))
        s = sn

    s = np.random.uniform(-0.05, 0.05, 4)
    s[0] = 0.0
    for _ in range(10):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        sn[0] = 0.0
        data.append((s.copy(), a, sn.copy(), 'x_fix'))
        s = sn

np.random.shuffle(data)
print(f'Total: {len(data)}')

S = np.array([d[0] for d in data])
A = np.array([d[1] for d in data])
SN = np.array([d[2] for d in data])
T = np.array([d[3] for d in data])

St = torch.FloatTensor(S)
At = torch.FloatTensor(A).float().unsqueeze(-1)
SNt = torch.FloatTensor(SN)

# Base model (from L2 v3 - best so far)
class AblationModel(nn.Module):
    def __init__(self, comp_weight=0.2, mult_dynamics=False, latent_dim=4):
        super().__init__()
        self.comp_weight = comp_weight
        self.mult_dynamics = mult_dynamics
        
        self.enc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        
        if mult_dynamics:
            # Multiplicative dynamics
            self.dyn = nn.Sequential(nn.Linear(latent_dim + 1, 32), nn.ReLU())
            self.dyn_out = nn.Linear(32, latent_dim)
        else:
            self.dyn = nn.Sequential(nn.Linear(latent_dim + 1, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        
        # Composition layer
        self.comp = nn.Sequential(nn.Linear(latent_dim * 2, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        
        self.dec = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 4))
    
    def forward(self, o, a):
        z = self.enc(o)
        
        if self.mult_dynamics:
            h = self.dyn(torch.cat([z, a], dim=-1))
            delta = torch.tanh(self.dyn_out(h)) * z  # multiplicative
        else:
            delta = self.dyn(torch.cat([z, a], dim=-1))
        
        z_next = z + delta
        
        # Composition
        z_pair = torch.cat([z, z_next], dim=-1)
        z_comp = self.comp(z_pair)
        
        z_final = z_next + self.comp_weight * z_comp
        
        recon = self.dec(z_final)
        
        return recon, z

def train_and_eval(comp_weight=0.2, mult_dynamics=False, latent_dim=4, epochs=10, lr=3e-4):
    model = AblationModel(comp_weight, mult_dynamics, latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            batch_idx = idx[i:i+32]
            pred, z = model(St[batch_idx], At[batch_idx])
            loss = F.mse_loss(pred, SNt[batch_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        _, z = model(St, At)
    
    z_np = z.numpy()
    
    # Variance analysis
    normal_idx = np.where(T == 'normal')[0]
    theta_idx = np.where(T == 'theta_fix')[0]
    
    z_normal = z_np[normal_idx]
    z_theta = z_np[theta_idx]
    
    std_normal = z_normal.std()
    std_theta = z_theta.std()
    
    reduction = (std_normal - std_theta) / std_normal * 100
    
    return reduction

# Ablation 1: Composition weight
print('\n' + '='*60)
print('1. COMPOSITION WEIGHT ABLATION')
print('='*60)

for cw in [0.0, 0.1, 0.2, 0.3, 0.5]:
    r = train_and_eval(comp_weight=cw)
    print(f'comp_weight={cw}: {r:+.1f}%')

# Ablation 2: Dynamics type
print('\n' + '='*60)
print('2. DYNAMICS TYPE ABLATION')
print('='*60)

for mult in [False, True]:
    r = train_and_eval(mult_dynamics=mult)
    print(f'multiplicative={mult}: {r:+.1f}%')

# Ablation 3: Latent dimension
print('\n' + '='*60)
print('3. LATENT DIMENSION ABLATION')
print('='*60)

for ld in [2, 4, 6, 8]:
    r = train_and_eval(latent_dim=ld)
    print(f'latent_dim={ld}: {r:+.1f}%')

# Ablation 4: Training epochs
print('\n' + '='*60)
print('4. EPOCHS ABLATION')
print('='*60)

for ep in [5, 10, 15, 20]:
    r = train_and_eval(epochs=ep)
    print(f'epochs={ep}: {r:+.1f}%')

# Ablation 5: Learning rate
print('\n' + '='*60)
print('5. LEARNING RATE ABLATION')
print('='*60)

for lr in [1e-4, 3e-4, 1e-3]:
    r = train_and_eval(lr=lr)
    print(f'lr={lr}: {r:+.1f}%')

print('\n' + '='*60)
print('DIAGNOSIS COMPLETE')
print('='*60)
