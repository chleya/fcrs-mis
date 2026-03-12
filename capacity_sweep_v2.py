"""Capacity sweep: latent_dim vs intervention_ratio"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

SEED = 42

def pend(s, a):
    t, o = s
    to = (a - 0.5) * 2
    on = o + (-9.8*np.sin(t) - 0.1*o + to) * 0.05
    tn = t + on * 0.05
    return np.array([tn, on])

class M(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.enc = nn.Sequential(nn.Linear(2, 24), nn.ReLU(), nn.Linear(24, dim))
        self.dyn = nn.Sequential(nn.Linear(dim+1, 24), nn.ReLU(), nn.Linear(24, dim))
        self.dec = nn.Sequential(nn.Linear(dim, 24), nn.ReLU(), nn.Linear(24, 2))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

def test_dim(dim):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    
    # Normal data
    data = []
    for _ in range(1500):
        s = np.random.uniform(-0.5, 0.5, 2)
        for _ in range(15):
            a = np.random.randint(0, 2)
            sn = pend(s, a)
            data.append((s.copy(), a, sn.copy(), 'n'))
            s = sn
    
    # Intervention (theta_fix)
    for _ in range(400):
        s = np.random.uniform(-0.5, 0.5, 2)
        s[0] = 0.0
        for _ in range(10):
            a = np.random.randint(0, 2)
            sn = pend(s, a)
            sn[0] = 0.0
            data.append((s.copy(), a, sn.copy(), 'i'))
            s = sn
    
    S = np.array([d[0] for d in data])
    A = np.array([d[1] for d in data])
    SN = np.array([d[2] for d in data])
    T = np.array([d[3] for d in data])
    
    St = torch.FloatTensor(S)
    At = torch.FloatTensor(A).float().unsqueeze(-1)
    SNt = torch.FloatTensor(SN)
    
    # Train
    m = M(dim)
    o = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            p, z = m(St[idx[i:i+32]], At[idx[i:i+32]])
            loss = F.mse_loss(p, SNt[idx[i:i+32]])
            loss.backward(); o.step(); o.zero_grad()
    
    m.eval()
    with torch.no_grad():
        p, z = m(St, At)
    
    # Correlation
    z_np = z.numpy()
    corr_theta = max(abs(np.corrcoef(z_np[:,0], S[:,0])[0,1]), 
                    abs(np.corrcoef(z_np[:,1], S[:,0])[0,1]) if dim>1 else 0)
    corr_omega = max(abs(np.corrcoef(z_np[:,0], S[:,1])[0,1]),
                     abs(np.corrcoef(z_np[:,1], S[:,1])[0,1]) if dim>1 else 0)
    
    # Intervention ratio
    ni, ii = T=='n', T=='i'
    ratio = np.mean((p.numpy()[ii]-SN[ii])**2) / np.mean((p.numpy()[ni]-SN[ni])**2)
    
    return {
        'dim': dim,
        'corr_theta': corr_theta,
        'corr_omega': corr_omega,
        'ratio': ratio
    }

print('='*60)
print('CAPACITY SWEEP: latent_dim vs intervention_ratio')
print('='*60)
print(f'{"dim":>4} | {"corr_theta":>10} | {"corr_omega":>10} | {"ratio":>8}')
print('-'*60)

dims = [2, 3, 4, 6, 8, 12, 16]
results = []

for d in dims:
    r = test_dim(d)
    results.append(r)
    print(f'{r["dim"]:>4} | {r["corr_theta"]:>10.3f} | {r["corr_omega"]:>10.3f} | {r["ratio"]:>8.3f}')

print('='*60)
print('SUMMARY')
print('='*60)

# Find optimal
best_ratio = min(results, key=lambda x: x['ratio'])
best_corr = max(results, key=lambda x: x['corr_theta'] + x['corr_omega'])

print(f'Best ratio: dim={best_ratio["dim"]}, ratio={best_ratio["ratio"]:.3f}')
print(f'Best corr:  dim={best_corr["dim"]}, corr={best_corr["corr_theta"]+best_corr["corr_omega"]:.3f}')

print('\n' + '='*60)
print('INTERPRETATION')
print('='*60)
print('If ratio < 1: intervention helps prediction')
print('If ratio > 1: intervention hurts prediction')
print('If corr ~ 1: latent captures true variables')
