"""
L2 Mechanism - Multiple Intervention Types with CORRECT METRIC
Test: Intervention MSE / Normal MSE ratio

Run: python l2_correct_metric.py
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

# Models
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, 4))
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1))

class Causal(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
    def forward(self, s, a):
        z = self.enc(s)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

# Test different interventions
def test_intervention(intervention_type='theta_fix'):
    # Generate data
    data = []
    
    # Normal
    for _ in range(1500):
        s = np.random.uniform(-0.05, 0.05, 4)
        for _ in range(15):
            a = np.random.randint(0, 2)
            sn = step(s, a)
            data.append((s.copy(), a, sn.copy(), 'normal'))
            s = sn
    
    # Intervention
    for _ in range(400):
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
                
        elif intervention_type == 'gaussian_noise':
            for _ in range(10):
                a = np.random.randint(0, 2)
                sn = step(s, a)
                sn += np.random.randn(4) * 0.5
                data.append((s.copy(), a, sn.copy(), 'int'))
                s = sn
                
        elif intervention_type == 'reverse_action':
            for _ in range(10):
                a = 1 - np.random.randint(0, 2)
                sn = step(s, a)
                data.append((s.copy(), a, sn.copy(), 'int'))
                s = sn
    
    S = np.array([d[0] for d in data])
    A = np.array([d[1] for d in data])
    SN = np.array([d[2] for d in data])
    T = np.array([d[3] for d in data])
    
    St = torch.FloatTensor(S)
    At = torch.FloatTensor(A).float().unsqueeze(-1)
    SNt = torch.FloatTensor(SN)
    
    # Train Baseline
    m1 = Baseline()
    opt = torch.optim.Adam(m1.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            pred = m1(St[idx[i:i+32]], At[idx[i:i+32]])
            loss = F.mse_loss(pred, SNt[idx[i:i+32]])
            loss.backward(); opt.step(); opt.zero_grad()
    
    m1.eval()
    with torch.no_grad():
        pred1 = m1(St, At)
    
    # Train Causal
    m2 = Causal()
    opt = torch.optim.Adam(m2.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            pred, _ = m2(St[idx[i:i+32]], At[idx[i:i+32]])
            loss = F.mse_loss(pred, SNt[idx[i:i+32]])
            loss.backward(); opt.step(); opt.zero_grad()
    
    m2.eval()
    with torch.no_grad():
        pred2, z = m2(St, At)
    
    # Evaluate
    ni = T == 'normal'
    ii = T == 'int'
    
    pred1_np = pred1.detach().numpy()
    pred2_np = pred2.detach().numpy()
    
    mse_n_base = np.mean((pred1_np[ni] - SN[ni])**2)
    mse_i_base = np.mean((pred1_np[ii] - SN[ii])**2)
    
    mse_n_causal = np.mean((pred2_np[ni] - SN[ni])**2)
    mse_i_causal = np.mean((pred2_np[ii] - SN[ii])**2)
    
    ratio_base = mse_i_base / mse_n_base
    ratio_causal = mse_i_causal / mse_n_causal
    
    # Correlation
    z_np = z.detach().numpy()
    corr_vel = np.corrcoef(z_np[:, 0], S[:, 1])[0, 1]
    
    return {
        'type': intervention_type,
        'base_ratio': ratio_base,
        'causal_ratio': ratio_causal,
        'improvement': (ratio_base - ratio_causal) / ratio_base * 100,
        'corr_vel': corr_vel
    }

# Test all interventions
interventions = ['theta_fix', 'x_fix', 'gaussian_noise', 'reverse_action']

print('='*60)
print('L2 MECHANISM - CORRECT METRIC')
print('='*60)

results = []

for int_type in interventions:
    print(f'\nTesting: {int_type}...')
    r = test_intervention(int_type)
    results.append(r)
    print(f'  Baseline ratio: {r["base_ratio"]:.3f}')
    print(f'  Causal ratio:   {r["causal_ratio"]:.3f}')
    print(f'  Improvement:    {r["improvement"]:+.1f}%')
    print(f'  Corr(z, vel):   {r["corr_vel"]:.3f}')

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f"{'Intervention':<20} | {'Baseline':>10} | {'Causal':>10} | {'Improv':>10}")
print('-'*60)
for r in results:
    print(f"{r['type']:<20} | {r['base_ratio']:>10.3f} | {r['causal_ratio']:>10.3f} | {r['improvement']:>+9.1f}%")

print('\n' + '='*60)
print('CONCLUSION')
print('='*60)
improvements = [r['improvement'] for r in results]
avg = np.mean(improvements)
print(f'Average improvement: {avg:+.1f}%')
if avg > 0:
    print('=> Causal consistently helps on intervention!')
else:
    print('=> Mixed results')
