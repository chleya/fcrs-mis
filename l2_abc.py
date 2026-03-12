"""
L2 Multi-seed + Pendulum + Visualization
Run: python l2_abc.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

SEED = 42

def step(s, a):
    x, xd, t, td = s
    f = 10.0 if a==1 else -10.0
    ct, st = np.cos(t), np.sin(t)
    tmp = f/1.1 + 0.05*td**2*st
    ta = (9.8*st - ct*tmp)/(0.5*(4/3-0.1*ct**2/1.1))
    return np.array([x+0.02*xd, xd+0.02*(tmp-0.05*ta*ct/1.1), t+0.02*td, td+0.02*ta])

def pendulum_step(s, a):
    theta, omega = s
    torque = (a - 0.5) * 2
    omega_next = omega + (-9.8*np.sin(theta) - 0.1*omega + torque) * 0.05
    theta_next = theta + omega_next * 0.05
    return np.array([theta_next, omega_next])

class Baseline(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim+1, 24), nn.ReLU(), nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, dim))
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1))

class Causal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, dim))
    def forward(self, s, a):
        z = self.enc(s)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

def test_cartpole(seed, int_type='x_fix'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    data = []
    for _ in range(1500):
        s = np.random.uniform(-0.05, 0.05, 4)
        for _ in range(15):
            a = np.random.randint(0, 2)
            sn = step(s, a)
            data.append((s.copy(), a, sn.copy(), 'n'))
            s = sn
    
    for _ in range(400):
        s = np.random.uniform(-0.05, 0.05, 4)
        if int_type == 'x_fix': s[0] = 0.0
        else: s[2] = 0.0
        for _ in range(10):
            a = np.random.randint(0, 2)
            sn = step(s, a)
            if int_type == 'x_fix': sn[0] = 0.0
            else: sn[2] = 0.0
            data.append((s.copy(), a, sn.copy(), 'i'))
            s = sn
    
    S = np.array([d[0] for d in data])
    A = np.array([d[1] for d in data])
    SN = np.array([d[2] for d in data])
    T = np.array([d[3] for d in data])
    
    St, At, SNt = map(torch.FloatTensor, [S, A, SN])
    
    # Baseline
    m1 = Baseline(4)
    o = torch.optim.Adam(m1.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            loss = F.mse_loss(m1(St[idx[i:i+32]], At[idx[i:i+32]].unsqueeze(-1)), SNt[idx[i:i+32]])
            loss.backward(); o.step(); o.zero_grad()
    
    m1.eval()
    with torch.no_grad():
        p1 = m1(St, At.unsqueeze(-1)).numpy()
    
    # Causal
    m2 = Causal(4)
    o = torch.optim.Adam(m2.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            p, _ = m2(St[idx[i:i+32]], At[idx[i:i+32]].unsqueeze(-1))
            loss = F.mse_loss(p, SNt[idx[i:i+32]])
            loss.backward(); o.step(); o.zero_grad()
    
    m2.eval()
    with torch.no_grad():
        p2, z = m2(St, At.unsqueeze(-1))
        p2 = p2.detach().numpy()
        z = z.detach().numpy()
    
    ni, ii = T=='n', T=='i'
    
    r_base = np.mean((p1[ii]-SN[ii])**2) / np.mean((p1[ni]-SN[ni])**2)
    r_causal = np.mean((p2[ii]-SN[ii])**2) / np.mean((p2[ni]-SN[ni])**2)
    
    corr = np.corrcoef(z[:,0], S[:,1])[0,1]
    
    return r_base, r_causal, corr

def test_pendulum(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    data = []
    for _ in range(1500):
        s = np.random.uniform(-0.5, 0.5, 2)
        for _ in range(15):
            a = np.random.randint(0, 2)
            sn = pendulum_step(s, a)
            data.append((s.copy(), a, sn.copy(), 'n'))
            s = sn
    
    for _ in range(400):
        s = np.random.uniform(-0.5, 0.5, 2)
        s[0] = 0.0
        for _ in range(10):
            a = np.random.randint(0, 2)
            sn = pendulum_step(s, a)
            sn[0] = 0.0
            data.append((s.copy(), a, sn.copy(), 'i'))
            s = sn
    
    S = np.array([d[0] for d in data])
    A = np.array([d[1] for d in data])
    SN = np.array([d[2] for d in data])
    T = np.array([d[3] for d in data])
    
    St, At, SNt = map(torch.FloatTensor, [S, A, SN])
    
    m1 = Baseline(2)
    o = torch.optim.Adam(m1.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            loss = F.mse_loss(m1(St[idx[i:i+32]], At[idx[i:i+32]].unsqueeze(-1)), SNt[idx[i:i+32]])
            loss.backward(); o.step(); o.zero_grad()
    
    m1.eval()
    with torch.no_grad():
        p1 = m1(St, At.unsqueeze(-1)).numpy()
    
    m2 = Causal(2)
    o = torch.optim.Adam(m2.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            p, _ = m2(St[idx[i:i+32]], At[idx[i:i+32]].unsqueeze(-1))
            loss = F.mse_loss(p, SNt[idx[i:i+32]])
            loss.backward(); o.step(); o.zero_grad()
    
    m2.eval()
    with torch.no_grad():
        p2, z = m2(St, At.unsqueeze(-1))
        p2 = p2.detach().numpy()
        z = z.detach().numpy()
    
    ni, ii = T=='n', T=='i'
    
    r_base = np.mean((p1[ii]-SN[ii])**2) / np.mean((p1[ni]-SN[ni])**2)
    r_causal = np.mean((p2[ii]-SN[ii])**2) / np.mean((p2[ni]-SN[ni])**2)
    
    corr = np.corrcoef(z[:,0], S[:,1])[0,1]
    
    return r_base, r_causal, corr

print('='*60)
print('A. MULTI-SEED VERIFICATION (CartPole)')
print('='*60)

print('\nCartPole x_fix:')
for s in range(5):
    rb, rc, c = test_cartpole(s, 'x_fix')
    print(f'  seed {s}: base={rb:.3f}, causal={rc:.3f}, corr={c:.3f}')

print('\nCartPole theta_fix:')
for s in range(5):
    rb, rc, c = test_cartpole(s, 'theta_fix')
    print(f'  seed {s}: base={rb:.3f}, causal={rc:.3f}, corr={c:.3f}')

print('\n' + '='*60)
print('C. PENDULUM VERIFICATION')
print('='*60)

for s in range(3):
    rb, rc, c = test_pendulum(s)
    print(f'  seed {s}: base={rb:.3f}, causal={rc:.3f}, corr={c:.3f}')

print('\n' + '='*60)
print('B. VISUALIZATION (Trajectory)')
print('='*60)

# Simple trajectory test
random.seed(42); np.random.seed(42); torch.manual_seed(42)

data = []
s = np.array([0.0, 0.0, 0.1, 0.0])
for _ in range(50):
    a = 1
    sn = step(s, a)
    data.append((s.copy(), a, sn.copy()))
    s = sn

S = np.array([d[0] for d in data])
A = np.array([d[1] for d in data])
SN = np.array([d[2] for d in data])

St, At, SNt = map(torch.FloatTensor, [S, A, SN])

m = Causal(4)
o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(10):
    for i in range(0, len(St), 8):
        p, z = m(St[i:i+8], At[i:i+8].unsqueeze(-1))
        loss = F.mse_loss(p, SNt[i:i+8])
        loss.backward(); o.step(); o.zero_grad()

m.eval()
with torch.no_grad():
    pred, z = m(St, At.unsqueeze(-1))
    pred = pred.detach().numpy()
    z = z.detach().numpy()

print('\nTrajectory (first 10 steps):')
print(f'{"Step":>5} | {"True_x":>8} | {"Pred_x":>8} | {"Error":>8}')
print('-'*45)
for i in range(10):
    err = abs(pred[i,0] - SN[i,0])
    print(f'{i:>5} | {SN[i,0]:>8.3f} | {pred[i,0]:>8.3f} | {err:>8.3f}')

print('\nLatent evolution (first 10 steps):')
print(f'{"Step":>5} | {"z[0]":>8} | {"z[1]":>8} | {"z[2]":>8}')
print('-'*40)
for i in range(10):
    print(f'{i:>5} | {z[i,0]:>8.3f} | {z[i,1]:>8.3f} | {z[i,2]:>8.3f}')

print('\n' + '='*60)
print('COMPLETE')
print('='*60)
