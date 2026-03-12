"""Pendulum multi-intervention test"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

SEED = 42

def pend(s, a, g=9.8):
    t, o = s
    to = (a - 0.5) * 2
    on = o + (-g*np.sin(t) - 0.1*o + to) * 0.05
    tn = t + on * 0.05
    return np.array([tn, on])

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 2))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

def test(intervention_type):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    
    data = []
    
    # Normal
    for _ in range(1500):
        s = np.random.uniform(-0.5, 0.5, 2)
        for _ in range(15):
            a = np.random.randint(0, 2)
            sn = pend(s, a)
            data.append((s.copy(), a, sn.copy(), 'n'))
            s = sn
    
    # Intervention
    for _ in range(400):
        s = np.random.uniform(-0.5, 0.5, 2)
        
        if intervention_type == 'theta_fix':
            s[0] = 0.0
            for _ in range(10):
                a = np.random.randint(0, 2)
                sn = pend(s, a)
                sn[0] = 0.0
                data.append((s.copy(), a, sn.copy(), 'i'))
                s = sn
                
        elif intervention_type == 'omega_fix':
            s[1] = 0.0
            for _ in range(10):
                a = np.random.randint(0, 2)
                sn = pend(s, a)
                sn[1] = 0.0
                data.append((s.copy(), a, sn.copy(), 'i'))
                s = sn
                
        elif intervention_type == 'torque_random':
            for _ in range(10):
                a = np.random.randint(0, 2)
                sn = pend(s, a)
                # Add extra random torque
                sn = sn + np.random.randn(2) * 0.3
                data.append((s.copy(), a, sn.copy(), 'i'))
                s = sn
                
        elif intervention_type == 'gravity_shift':
            g = 15.0  # Higher gravity
            for _ in range(10):
                a = np.random.randint(0, 2)
                sn = pend(s, a, g)
                data.append((s.copy(), a, sn.copy(), 'i'))
                s = sn
    
    S = np.array([d[0] for d in data])
    A = np.array([d[1] for d in data])
    SN = np.array([d[2] for d in data])
    T = np.array([d[3] for d in data])
    
    St = torch.FloatTensor(S)
    At = torch.FloatTensor(A).float().unsqueeze(-1)
    SNt = torch.FloatTensor(SN)
    
    m = M()
    o = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            p, z = m(St[idx[i:i+32]], At[idx[i:i+32]])
            F.mse_loss(p, SNt[idx[i:i+32]]).backward()
            o.step(); o.zero_grad()
    
    m.eval()
    with torch.no_grad():
        p, z = m(St, At)
    
    ni, ii = T=='n', T=='i'
    
    r = np.mean((p.numpy()[ii]-SN[ii])**2) / np.mean((p.numpy()[ni]-SN[ni])**2)
    c = np.corrcoef(z.numpy()[:,0], S[:,0])[0,1]
    
    return r, c

print('='*60)
print('PENDULUM MULTI-INTERVENTION')
print('='*60)

types = ['theta_fix', 'omega_fix', 'torque_random', 'gravity_shift']

for t in types:
    r, c = test(t)
    print(f'{t:20} | ratio={r:.3f} | corr={c:.3f}')

print('='*60)
