"""Multi-seed verification: 10 seeds per capacity - FIXED"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def pend(s, a):
    t, o = s
    to = (a - 0.5) * 2
    on = o + (-9.8*np.sin(t) - 0.1*o + to) * 0.05
    tn = t + on * 0.05
    return np.array([tn, on])

class M(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, 24), nn.ReLU(), nn.Linear(24, dim))
        self.dyn = nn.Sequential(nn.Linear(dim+1, 24), nn.ReLU(), nn.Linear(24, dim))
        self.dec = nn.Sequential(nn.Linear(dim, 24), nn.ReLU(), nn.Linear(24, 2))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

def test_dim_seed(dim, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    # Generate data
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
    
    # Correlation with omega (the stronger signal)
    z_np = z.numpy()
    corrs = []
    for i in range(min(dim, z_np.shape[1])):
        c = np.corrcoef(z_np[:,i], S[:,1])[0,1]
        if not np.isnan(c):
            corrs.append(abs(c))
    corr_omega = max(corrs) if corrs else 0.0
    
    # Intervention ratio
    ni, ii = T=='n', T=='i'
    ratio = np.mean((p.numpy()[ii]-SN[ii])**2) / np.mean((p.numpy()[ni]-SN[ni])**2)
    
    return corr_omega, ratio

dims = [2, 3, 4, 6, 8]
seeds = list(range(10))

print('='*70)
print('MULTI-SEED VERIFICATION (10 seeds)')
print('='*70)

results = {d: {'corr': [], 'ratio': []} for d in dims}

for dim in dims:
    print(f'Testing dim={dim}...', end=' ')
    for seed in seeds:
        corr, ratio = test_dim_seed(dim, seed)
        results[dim]['corr'].append(corr)
        results[dim]['ratio'].append(ratio)
    print('done')

print('\n' + '='*70)
print('RESULTS SUMMARY')
print('='*70)
print(f'{"dim":>4} | {"corr_mean":>10} | {"corr_std":>10} | {"ratio_mean":>10} | {"ratio_std":>10}')
print('-'*70)

for dim in dims:
    c_mean = np.mean(results[dim]['corr'])
    c_std = np.std(results[dim]['corr'])
    r_mean = np.mean(results[dim]['ratio'])
    r_std = np.std(results[dim]['ratio'])
    print(f'{dim:>4} | {c_mean:>10.3f} | {c_std:>10.3f} | {r_mean:>10.3f} | {r_std:>10.3f}')

print('\n' + '='*70)
print('INTERPRETATION')
print('='*70)
print('Best ratio (lower is better):')
best_dim = min(dims, key=lambda d: np.mean(results[d]['ratio']))
print(f'  dim={best_dim}, ratio={np.mean(results[best_dim]["ratio"]):.3f}±{np.std(results[best_dim]["ratio"]):.3f}')
print('Best corr (higher is better):')
best_corr_dim = max(dims, key=lambda d: np.mean(results[d]['corr']))
print(f'  dim={best_corr_dim}, corr={np.mean(results[best_corr_dim]["corr"]):.3f}±{np.std(results[best_corr_dim]["corr"]):.3f}')
