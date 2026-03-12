"""
L2 Noise Type Ablation
Test different noise types for mechanism formation
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

def generate_data(noise_type='gaussian', noise_strength=0.5, n_normal=1500, n_int=400):
    data = []
    
    # Normal
    for _ in range(n_normal):
        s = np.random.uniform(-0.05, 0.05, 4)
        for _ in range(15):
            a = np.random.randint(0, 2)
            sn = step(s, a)
            data.append((s.copy(), a, sn.copy(), 'normal'))
            s = sn
    
    # Intervention
    for _ in range(n_int):
        s = np.random.uniform(-0.05, 0.05, 4)
        for _ in range(10):
            a = np.random.randint(0, 2)
            sn = step(s, a)
            
            if noise_type == 'gaussian':
                sn += np.random.randn(4) * noise_strength
            elif noise_type == 'uniform':
                sn += np.random.uniform(-1, 1, 4) * noise_strength
            elif noise_type == 'action':
                # Noise proportional to action
                sn += (a - 0.5) * noise_strength * 2
            elif noise_type == 'state':
                # Noise proportional to state
                sn += s * noise_strength
            elif noise_type == 'sine':
                # Periodic noise
                sn += np.sin(np.arange(4) * np.pi) * noise_strength
            
            data.append((s.copy(), a, sn.copy(), 'int'))
            s = sn
    
    return data

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

# Test noise types
noise_types = ['gaussian', 'uniform', 'action', 'state', 'sine', 'none']

print('='*60)
print('NOISE TYPE ABLATION')
print('='*60)

results = []

for noise_type in noise_types:
    print(f'\nTesting: {noise_type}')
    
    data = generate_data(noise_type=noise_type, noise_strength=0.5)
    S = np.array([d[0] for d in data])
    A = np.array([d[1] for d in data])
    SN = np.array([d[2] for d in data])
    T = np.array([d[3] for d in data])
    
    St = torch.FloatTensor(S)
    At = torch.FloatTensor(A).float().unsqueeze(-1)
    SNt = torch.FloatTensor(SN)
    
    # Train
    m = M()
    o = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(10):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            p, z = m(St[idx[i:i+32]], At[idx[i:i+32]])
            F.mse_loss(p, SNt[idx[i:i+32]]).backward()
            o.step(); o.zero_grad()
    
    m.eval()
    with torch.no_grad():
        _, z = m(St, At)
    
    ni = T=='normal'
    ii = T=='int'
    
    r = (z[ni].std() - z[ii].std()) / z[ni].std() * 100
    
    results.append({'type': noise_type, 'variance_change': r})
    print(f'  Variance change: {r:+.1f}%')

print('\n' + '='*60)
print('SUMMARY')
print('='*60)

for r in sorted(results, key=lambda x: x['variance_change']):
    print(f'{r["type"]:<10}: {r["variance_change"]:+.1f}%')

best = min(results, key=lambda x: x['variance_change'])
print(f'\nBest noise type: {best["type"]} ({best["variance_change"]:+.1f}%)')
