"""
Quantify L2 Mechanism Strength
Calculate:
1. Information flow: MI(z; state) before/after noise
2. Causal strength: intervention effect size
3. Variance reduction statistics

Run: python l2_quantify.py
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

def step(s, a):
    x, xd, t, td = s
    f = 10.0 if a==1 else -10.0
    ct, st = np.cos(t), np.sin(t)
    tmp = f/1.1 + 0.05*td**2*st
    ta = (9.8*st - ct*tmp)/(0.5*(4/3-0.1*ct**2/1.1))
    return np.array([x+0.02*xd, xd+0.02*(tmp-0.05*ta*ct/1.1), t+0.02*td, td+0.02*ta])

# Data
print('Generating data...')
data = []
for _ in range(1500):
    s = np.random.uniform(-0.05, 0.05, 4)
    for _ in range(15):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        data.append((s.copy(), a, sn.copy(), 'normal'))
        s = sn

for _ in range(400):
    s = np.random.uniform(-0.05, 0.05, 4)
    for _ in range(10):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        sn += np.random.randn(4) * 0.5
        data.append((s.copy(), a, sn.copy(), 'noise'))
        s = sn

S = np.array([d[0] for d in data])
A = np.array([d[1] for d in data])
SN = np.array([d[2] for d in data])
T = np.array([d[3] for d in data])

St = torch.FloatTensor(S)
At = torch.FloatTensor(A).float().unsqueeze(-1)
SNt = torch.FloatTensor(SN)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z,a],-1))
        return self.dec(zn), z

# Train
print('\nTraining...')
m = M()
o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(10):
    idx = np.random.permutation(len(St))
    for i in range(0, len(idx), 32):
        batch_idx = idx[i:i+32]
        p, z = m(St[batch_idx], At[batch_idx])
        F.mse_loss(p, SNt[batch_idx]).backward()
        o.step(); o.zero_grad()

# Evaluate
m.eval()
with torch.no_grad():
    _, z = m(St, At)

z_np = z.numpy()

# Analysis
normal_idx = T == 'normal'
noise_idx = T == 'noise'

z_normal = z_np[normal_idx]
z_noise = z_np[noise_idx]
s_normal = S[normal_idx]
s_noise = S[noise_idx]

print('\n' + '='*60)
print('L2 MECHANISM QUANTIFICATION')
print('='*60)

# 1. Variance analysis
print('\n1. VARIANCE ANALYSIS')
print('-'*40)
print(f'Normal:  mean={z_normal.mean():.3f}, std={z_normal.std():.3f}')
print(f'Noise:   mean={z_noise.mean():.3f}, std={z_noise.std():.3f}')

variance_reduction = (z_normal.std() - z_noise.std()) / z_normal.std() * 100
print(f'Variance reduction: {variance_reduction:+.1f}%')

# 2. Correlation with states
print('\n2. LATENT-STATE CORRELATIONS')
print('-'*40)
for i in range(4):
    for j, name in enumerate(['x', 'xd', 'theta', 'td']):
        c = np.corrcoef(z_np[:, i], S[:, j])[0, 1]
        if not np.isnan(c):
            print(f'z[{i}]-{name}: {c:+.3f}')

# 3. Information-theoretic: variance of latent means
print('\n3. LATENT MEAN STABILITY')
print('-'*40)
# Compare latent distributions
mean_diff = np.abs(z_normal.mean(axis=0) - z_noise.mean(axis=0))
print(f'Mean difference (normal vs noise): {mean_diff}')
print(f'Total mean shift: {mean_diff.sum():.3f}')

# 4. Effect size (Cohen's d)
print('\n4. EFFECT SIZE (Cohen d)')
print('-'*40)
pooled_std = np.sqrt((z_normal.std()**2 + z_noise.std()**2) / 2)
cohens_d = (z_normal.mean() - z_noise.mean()) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")
if abs(cohens_d) > 0.8:
    print('=> Large effect size!')

# 5. Latent dimension analysis
print('\n5. PER-LATENT VARIANCE')
print('-'*40)
for i in range(4):
    zn_i = z_normal[:, i]
    zi_i = z_noise[:, i]
    vr_i = (zn_i.std() - zi_i.std()) / zn_i.std() * 100
    print(f'z[{i}]: normal={zn_i.std():.3f}, noise={zi_i.std():.3f}, reduction={vr_i:+.1f}%')

# 6. Summary metrics
print('\n' + '='*60)
print('SUMMARY METRICS')
print('='*60)
print(f'Variance Reduction:    {variance_reduction:+.1f}%')
print(f"Cohen's d:            {cohens_d:.3f}")
print(f'Mean Shift:           {mean_diff.sum():.3f}')

# Interpret
print('\n' + '='*60)
print('INTERPRETATION')
print('='*60)
if variance_reduction < -50:
    print('=> STRONG mechanism formation!')
elif variance_reduction < 0:
    print('=> WEAK mechanism detected')
else:
    print('=> No mechanism')
