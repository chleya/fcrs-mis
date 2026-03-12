"""
L2 v3 - Strong Composition Loss + Multi-step Intervention
Tests whether composition loss can trigger mechanism formation

Key upgrades:
1. Composition loss: encourage z[0] * z[1] binding
2. Multi-step intervention: fix z over multiple steps
3. Causal chain loss

Run: python l2_v3.py
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
print('L2 V3 - STRONG COMPOSITION MECHANISM')
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

# Add multi-step interventions
print('Adding interventions...')
for _ in range(500):
    # Fix theta = 0 for 10 steps
    s = np.random.uniform(-0.05, 0.05, 4)
    s[2] = 0.0  # theta = 0
    for _ in range(10):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        sn[2] = 0.0  # keep fixed
        data.append((s.copy(), a, sn.copy(), 'theta_fix'))
        s = sn
    
    # Fix x = 0 for 10 steps
    s = np.random.uniform(-0.05, 0.05, 4)
    s[0] = 0.0
    for _ in range(10):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        sn[0] = 0.0
        data.append((s.copy(), a, sn.copy(), 'x_fix'))
        s = sn

np.random.shuffle(data)
print(f'Total samples: {len(data)}')

# Split by type
normal_data = [d for d in data if d[3] == 'normal']
theta_data = [d for d in data if d[3] == 'theta_fix']
x_data = [d for d in data if d[3] == 'x_fix']

S = np.array([d[0] for d in data])
A = np.array([d[1] for d in data])
SN = np.array([d[2] for d in data])

St = torch.FloatTensor(S)
At = torch.FloatTensor(A).float().unsqueeze(-1)
SNt = torch.FloatTensor(SN)

# Model with composition loss
class CompositionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 4))
        
        # Composition layer - bind z[0] and z[1]
        self.comp = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 4))
        
        self.dec = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 4))
    
    def forward(self, o, a):
        z = self.enc(o)
        
        # Dynamics
        delta = self.dyn(torch.cat([z, a], dim=-1))
        z_next = z + delta
        
        # Composition: encourage z[0] * z[1] binding
        # z_pair: [z0, z1, z0_next, z1_next]
        z_pair = torch.cat([z[:, 0:2], z_next[:, 0:2]], dim=-1)  # (batch, 4)
        z_comp = self.comp(z_pair)
        
        # Combine
        z_final = z_next + 0.1 * z_comp
        
        recon = self.dec(z_final)
        
        return recon, z, z_comp

# Train with composition loss
print('\nTraining...')
model = CompositionModel()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(15):
    idx = np.random.permutation(len(St))
    for i in range(0, len(idx), 32):
        batch_idx = idx[i:i+32]
        o = St[batch_idx]
        a = At[batch_idx]
        s_next = SNt[batch_idx]
        
        pred, z, z_comp = model(o, a)
        
        # Reconstruction loss
        loss_rec = F.mse_loss(pred, s_next)
        
        # Composition loss: encourage independence + binding
        # Try to make z_comp small (composition shouldn't dominate)
        loss_comp = F.mse_loss(z_comp, torch.zeros_like(z_comp))
        
        # Total loss
        loss = loss_rec + 0.3 * loss_comp
        
        opt.zero_grad()
        loss.backward()
        opt.step()

# Evaluate
model.eval()
with torch.no_grad():
    pred, z, z_comp = model(St, At)

z_np = z.numpy()

print('\n' + '='*60)
print('RESULTS')
print('='*60)

# Correlations
print('\nCorrelations:')
for i in range(4):
    for j, n in enumerate(['x','xd','t','td']):
        c = np.corrcoef(z_np[:, i], S[:, j])[0, 1]
        if not np.isnan(c):
            print(f'z[{i}]-{n}: {c:+.3f}')

# Variance analysis
normal_idx = [i for i, d in enumerate(data) if d[3] == 'normal']
theta_idx = [i for i, d in enumerate(data) if d[3] == 'theta_fix']
x_idx = [i for i, d in enumerate(data) if d[3] == 'x_fix']

z_normal = z_np[normal_idx]
z_theta = z_np[theta_idx]
z_x = z_np[x_idx]

print('\n' + '='*60)
print('VARIANCE ANALYSIS')
print('='*60)
print(f'Normal:    std = {z_normal.std():.3f}')
print(f'Theta fix: std = {z_theta.std():.3f}')
print(f'X fix:     std = {z_x.std():.3f}')

theta_reduction = (z_normal.std() - z_theta.std()) / z_normal.std() * 100
x_reduction = (z_normal.std() - z_x.std()) / z_normal.std() * 100

print(f'\nTheta intervention reduction: {theta_reduction:+.1f}%')
print(f'X intervention reduction:     {x_reduction:+.1f}%')

print('\n' + '='*60)
print('CONCLUSION')
print('='*60)
if theta_reduction >= 30 or x_reduction >= 30:
    print('=> MECHANISM DETECTED! >= 30% reduction')
elif theta_reduction >= 15 or x_reduction >= 15:
    print('=> PARTIAL mechanism (15-30%)')
else:
    print('=> Still weak mechanism (<15%)')
    print('=> Consider: stronger composition or different architecture')
