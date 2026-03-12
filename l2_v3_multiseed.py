"""
L2 v3 Multi-Seed Verification
Run L2 v3 with 10 seeds to verify -27.1% is reproducible

Run: python l2_v3_multiseed.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# CartPole
def step(s, a):
    x, xd, t, td = s
    f = 10.0 if a==1 else -10.0
    ct, st = np.cos(t), np.sin(t)
    tmp = f/1.1 + 0.05*td**2*st
    ta = (9.8*st - ct*tmp)/(0.5*(4/3-0.1*ct**2/1.1))
    xa = tmp - 0.05*ta*ct/1.1
    return np.array([x+0.02*xd, xd+0.02*xa, t+0.02*td, td+0.02*ta])

# Generate fixed data
print('Generating data...')
data = []
for _ in range(2000):
    s = np.random.uniform(-0.05, 0.05, 4)
    for _ in range(20):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        data.append((s.copy(), a, sn.copy(), 'normal'))
        s = sn

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

S = np.array([d[0] for d in data])
A = np.array([d[1] for d in data])
SN = np.array([d[2] for d in data])
T = np.array([d[3] for d in data])

St = torch.FloatTensor(S)
At = torch.FloatTensor(A).float().unsqueeze(-1)
SNt = torch.FloatTensor(SN)

# L2 v3 model (best architecture)
class L2V3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 4))
        self.comp = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 4))
        self.dec = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 4))
    
    def forward(self, o, a):
        z = self.enc(o)
        delta = self.dyn(torch.cat([z, a], dim=-1))
        z_next = z + delta
        z_pair = torch.cat([z, z_next], dim=-1)
        z_comp = self.comp(z_pair)
        z_final = z_next + 0.1 * z_comp
        return self.dec(z_final), z

# Run with multiple seeds
num_seeds = 10
results = []

print('\n' + '='*60)
print(f'L2 V3 MULTI-SEED VERIFICATION ({num_seeds} seeds)')
print('='*60)

for seed in range(num_seeds):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = L2V3Model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(12):
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
    
    normal_idx = np.where(T == 'normal')[0]
    theta_idx = np.where(T == 'theta_fix')[0]
    x_idx = np.where(T == 'x_fix')[0]
    
    z_normal = z_np[normal_idx]
    z_theta = z_np[theta_idx]
    z_x = z_np[x_idx]
    
    std_normal = z_normal.std()
    std_theta = z_theta.std()
    std_x = z_x.std()
    
    reduction_theta = (std_normal - std_theta) / std_normal * 100
    reduction_x = (std_normal - std_x) / std_normal * 100
    mean_reduction = (reduction_theta + reduction_x) / 2
    
    results.append({
        'seed': seed,
        'std_normal': std_normal,
        'std_theta': std_theta,
        'std_x': std_x,
        'reduction_theta': reduction_theta,
        'reduction_x': reduction_x,
        'mean_reduction': mean_reduction
    })
    
    print(f'Seed {seed}: mean reduction = {mean_reduction:+.1f}%')

# Summary
print('\n' + '='*60)
print('SUMMARY')
print('='*60)

reductions = [r['mean_reduction'] for r in results]
mean = np.mean(reductions)
std = np.std(reductions)
min_r = np.min(reductions)
max_r = np.max(reductions)

print(f'Mean reduction: {mean:+.1f}%')
print(f'Std: {std:.1f}%')
print(f'Range: [{min_r:+.1f}%, {max_r:+.1f}%]')
print(f'Seeds with negative (mechanism): {sum(1 for r in reductions if r < 0)}/{num_seeds}')

print('\n' + '='*60)
print('CONCLUSION')
print('='*60)
if mean < -20 and std < 15:
    print('=> ROBUST mechanism detected!')
elif mean < 0:
    print('=> WEAK mechanism (negative mean)')
else:
    print('=> NO mechanism - variance increases on average')
    print(f'   The -27.1% result from v3 is likely seed-dependent')
