"""Trajectory visualization for CartPole"""
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

class Causal(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

# Generate trajectory
print('Generating trajectory...')
traj = []
s = np.array([0.0, 0.0, 0.1, 0.0])
for _ in range(50):
    a = 1
    sn = step(s, a)
    traj.append({'s': s.copy(), 'a': a, 'sn': sn.copy()})
    s = sn

S = np.array([d['s'] for d in traj])
A = np.array([d['a'] for d in traj])
SN = np.array([d['sn'] for d in traj])

St = torch.FloatTensor(S)
At = torch.FloatTensor(A).float().unsqueeze(-1)
SNt = torch.FloatTensor(SN)

# Train
m = Causal()
o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(15):
    for i in range(0, len(St), 8):
        p, z = m(St[i:i+8], At[i:i+8])
        loss = F.mse_loss(p, SNt[i:i+8])
        loss.backward(); o.step(); o.zero_grad()

m.eval()
with torch.no_grad():
    pred, z = m(St, At)
    pred = pred.numpy()
    z = z.numpy()

# Print trajectory
print('\n' + '='*70)
print('TRAJECTORY VISUALIZATION')
print('='*70)

print('\n1. STATE VARIABLES:')
print('='*70)
print(f'{"Step":>5} | {"x":>8} | {"xd":>8} | {"theta":>8} | {"td":>8}')
print('-'*70)
for i in range(min(25, len(S))):
    print(f'{i:>5} | {S[i,0]:>8.3f} | {S[i,1]:>8.3f} | {S[i,2]:>8.3f} | {S[i,3]:>8.3f}')

print('\n2. LATENT VARIABLES:')
print('='*70)
print(f'{"Step":>5} | {"z0":>8} | {"z1":>8} | {"z2":>8} | {"z3":>8}')
print('-'*70)
for i in range(min(25, len(z))):
    print(f'{i:>5} | {z[i,0]:>8.3f} | {z[i,1]:>8.3f} | {z[i,2]:>8.3f} | {z[i,3]:>8.3f}')

print('\n3. PREDICTION vs TRUE:')
print('='*70)
print(f'{"Step":>5} | {"True_x":>8} | {"Pred_x":>8} | {"Error":>8}')
print('-'*70)
for i in range(min(25, len(pred))):
    err = abs(pred[i,0] - SN[i,0])
    print(f'{i:>5} | {SN[i,0]:>8.3f} | {pred[i,0]:>8.3f} | {err:>8.3f}')

print('\n4. CORRELATION ANALYSIS:')
print('='*70)
for i in range(4):
    for j, name in enumerate(['x', 'xd', 'theta', 'td']):
        c = np.corrcoef(z[:, i], S[:, j])[0, 1]
        if not np.isnan(c):
            print(f'z[{i}] vs {name}: {c:+.3f}')

print('\n' + '='*70)
print('VISUAL SUMMARY')
print('='*70)
print('If z[i] closely tracks a state variable, correlation should be >0.9')
