"""
L2 Intervention + Temporal Ablation
Test different intervention types + temporal consistency

Run: python l2_intervention_temporal.py
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
    xa = tmp - 0.05*ta*ct/1.1
    return np.array([x+0.02*xd, xd+0.02*xa, t+0.02*td, td+0.02*ta])

# Generate data with different intervention types
print('='*60)
print('INTERVENTION TYPE ABLATION')
print('='*60)

def generate_data(intervention_type='theta_fix', n_normal=1500, n_int=300):
    """Types: theta_fix, x_fix, both_fix, random_noise, reverse_action"""
    data = []
    
    # Normal
    for _ in range(n_normal):
        s = np.random.uniform(-0.05, 0.05, 4)
        for _ in range(15):
            a = np.random.randint(0, 2)
            sn = step(s, a)
            data.append((s.copy(), a, sn.copy(), 'normal'))
            s = sn
    
    # Interventions
    for _ in range(n_int):
        s = np.random.uniform(-0.05, 0.05, 4)
        
        if intervention_type == 'theta_fix':
            s[2] = 0.0
            for _ in range(10):
                a = np.random.randint(0, 2)
                sn = step(s, a)
                sn[2] = 0.0
                data.append((s.copy(), a, sn.copy(), 'int'))
                s = sn
                
        elif intervention_type == 'x_fix':
            s[0] = 0.0
            for _ in range(10):
                a = np.random.randint(0, 2)
                sn = step(s, a)
                sn[0] = 0.0
                data.append((s.copy(), a, sn.copy(), 'int'))
                s = sn
                
        elif intervention_type == 'both_fix':
            s[0] = 0.0; s[2] = 0.0
            for _ in range(10):
                a = np.random.randint(0, 2)
                sn = step(s, a)
                sn[0] = 0.0; sn[2] = 0.0
                data.append((s.copy(), a, sn.copy(), 'int'))
                s = sn
                
        elif intervention_type == 'random_noise':
            for _ in range(10):
                a = np.random.randint(0, 2)
                sn = step(s, a)
                sn += np.random.randn(4) * 0.5  # Add noise
                data.append((s.copy(), a, sn.copy(), 'int'))
                s = sn
                
        elif intervention_type == 'reverse_action':
            for _ in range(10):
                a = 1 - np.random.randint(0, 2)  # Reverse action
                sn = step(s, a)
                data.append((s.copy(), a, sn.copy(), 'int'))
                s = sn
    
    return data

# Test different interventions
intervention_types = ['theta_fix', 'x_fix', 'both_fix', 'random_noise', 'reverse_action']

print('\n1. INTERVENTION TYPE ABLATION')
print('-'*50)

class Model(nn.Module):
    def __init__(self, use_temporal=False):
        super().__init__()
        self.use_temporal = use_temporal
        self.enc = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        
        if use_temporal:
            self.temp = nn.Sequential(nn.Linear(8, 24), nn.ReLU(), nn.Linear(24, 4))
        
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
    
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], -1))
        
        if self.use_temporal:
            zt = self.temp(torch.cat([z, zn], -1))
            zn = zn + 0.1 * zt
        
        return self.dec(zn), z

results = []

for int_type in intervention_types:
    print(f'\nTesting: {int_type}')
    
    data = generate_data(int_type)
    S = np.array([d[0] for d in data])
    A = np.array([d[1] for d in data])
    SN = np.array([d[2] for d in data])
    T = np.array([d[3] for d in data])
    
    St = torch.FloatTensor(S)
    At = torch.FloatTensor(A).float().unsqueeze(-1)
    SNt = torch.FloatTensor(SN)
    
    # Train without temporal
    m = Model(use_temporal=False)
    o = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            p, z = m(St[idx[i:i+32]], At[idx[i:i+32]])
            F.mse_loss(p, SNt[idx[i:i+32]]).backward()
            o.step(); o.zero_grad()
    
    with torch.no_grad():
        _, z = m(St, At)
    
    ni = T=='normal'; ii = T=='int'
    r_no_temp = (z[ni].std() - z[ii].std()) / z[ni].std() * 100
    
    # Train with temporal
    m = Model(use_temporal=True)
    o = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            p, z = m(St[idx[i:i+32]], At[idx[i:i+32]])
            F.mse_loss(p, SNt[idx[i:i+32]]).backward()
            o.step(); o.zero_grad()
    
    with torch.no_grad():
        _, z = m(St, At)
    
    r_with_temp = (z[ni].std() - z[ii].std()) / z[ni].std() * 100
    
    results.append({
        'type': int_type,
        'no_temp': r_no_temp,
        'with_temp': r_with_temp
    })
    
    print(f'  No temporal: {r_no_temp:+.1f}%')
    print(f'  With temporal: {r_with_temp:+.1f}%')

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f"{'Type':<15} | {'No Temp':>10} | {'With Temp':>10}")
print('-'*50)
for r in results:
    print(f"{r['type']:<15} | {r['no_temp']:>+10.1f} | {r['with_temp']:>+10.1f}")

# Best config
best = min(results, key=lambda x: x['no_temp'])
print(f'\nBest intervention: {best["type"]} ({best["no_temp"]:+.1f}%)')
