"""Test latent_dim=2 for Pendulum"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def pend(s, a):
    t, o = s
    to = (a - 0.5) * 2
    on = o + (-9.8*np.sin(t) - 0.1*o + to) * 0.05
    tn = t + on * 0.05
    return np.array([tn, on])

# Model with latent_dim=2
class M2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, 24), nn.ReLU(), nn.Linear(24, 2))
        self.dyn = nn.Sequential(nn.Linear(3, 24), nn.ReLU(), nn.Linear(24, 2))
        self.dec = nn.Sequential(nn.Linear(2, 24), nn.ReLU(), nn.Linear(24, 2))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

# Generate data
print('Generating data...')
data = []
for _ in range(1500):
    s = np.random.uniform(-0.5, 0.5, 2)
    for _ in range(15):
        a = np.random.randint(0, 2)
        sn = pend(s, a)
        data.append((s.copy(), a, sn.copy()))
        s = sn

S = np.array([d[0] for d in data])
A = np.array([d[1] for d in data])
SN = np.array([d[2] for d in data])

St = torch.FloatTensor(S)
At = torch.FloatTensor(A).float().unsqueeze(-1)
SNt = torch.FloatTensor(SN)

# Train
m = M2()
o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(10):
    idx = np.random.permutation(len(St))
    for i in range(0, len(idx), 32):
        p, z = m(St[idx[i:i+32]], At[idx[i:i+32]])
        loss = F.mse_loss(p, SNt[idx[i:i+32]])
        loss.backward(); o.step(); o.zero_grad()

m.eval()
with torch.no_grad():
    p, z = m(St, At)
    z = z.numpy()

# Correlation
print('\n' + '='*50)
print('latent_dim=2 RESULTS')
print('='*50)
for i in range(2):
    for j, name in enumerate(['theta', 'omega']):
        c = np.corrcoef(z[:, i], S[:, j])[0, 1]
        if not np.isnan(c):
            print(f'z[{i}] vs {name}: {c:+.3f}')

# Test intervention
print('\n' + '='*50)
print('INTERVENTION TEST')
print('='*50)

# Normal
n_data = []
for _ in range(500):
    s = np.random.uniform(-0.5, 0.5, 2)
    for _ in range(10):
        a = np.random.randint(0, 2)
        sn = pend(s, a)
        n_data.append((s.copy(), a, sn.copy(), 'n'))
        s = sn

# Theta fix
i_data = []
for _ in range(200):
    s = np.random.uniform(-0.5, 0.5, 2)
    s[0] = 0.0
    for _ in range(10):
        a = np.random.randint(0, 2)
        sn = pend(s, a)
        sn[0] = 0.0
        i_data.append((s.copy(), a, sn.copy(), 'i'))
        s = sn

ALL = n_data + i_data
S = np.array([d[0] for d in ALL])
A = np.array([d[1] for d in ALL])
SN = np.array([d[2] for d in ALL])
T = np.array([d[3] for d in ALL])

St = torch.FloatTensor(S)
At = torch.FloatTensor(A).float().unsqueeze(-1)
SNt = torch.FloatTensor(SN)

m = M2()
o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(10):
    idx = np.random.permutation(len(St))
    for i in range(0, len(idx), 32):
        p, z = m(St[idx[i:i+32]], At[idx[i:i+32]])
        loss = F.mse_loss(p, SNt[idx[i:i+32]])
        loss.backward(); o.step(); o.zero_grad()

m.eval()
with torch.no_grad():
    p = m(St, At)[0].numpy()

ni, ii = T=='n', T=='i'
r = np.mean((p[ii]-SN[ii])**2) / np.mean((p[ni]-SN[ni])**2)

print(f'Normal MSE: {np.mean((p[ni]-SN[ni])**2):.6f}')
print(f'Intervention MSE: {np.mean((p[ii]-SN[ii])**2):.6f}')
print(f'Ratio: {r:.3f}')
