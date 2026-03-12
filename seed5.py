"""Quick 5-seed test"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def step(s, a):
    x, xd, t, td = s
    f = 10.0 if a==1 else -10.0
    ct, st = np.cos(t), np.sin(t)
    tmp = f/1.1 + 0.05*td**2*st
    ta = (9.8*st - ct*tmp)/(0.5*(4/3-0.1*ct**2/1.1))
    return np.array([x+0.02*xd, xd+0.02*(tmp-0.05*ta*ct/1.1), t+0.02*td, td+0.02*ta])

# Data
data = []
for _ in range(1500):
    s = np.random.uniform(-0.05, 0.05, 4)
    for _ in range(15):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        data.append((s.copy(), a, sn.copy(), 'n'))
        s = sn

for _ in range(300):
    s = np.random.uniform(-0.05, 0.05, 4); s[2]=0.0
    for _ in range(8):
        a = np.random.randint(0, 2)
        sn = step(s, a); sn[2]=0.0
        data.append((s.copy(), a, sn.copy(), 't'))
        s = sn

S = np.array([d[0] for d in data])
A = np.array([d[1] for d in data])
SN = np.array([d[2] for d in data])
T = np.array([d[3] for d in data])
St, At, SNt = map(torch.FloatTensor, [S, A, SN])

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.comp = nn.Sequential(nn.Linear(8, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z,a],-1))
        zc = self.comp(torch.cat([z, zn], -1))
        return self.dec(zn + 0.1*zc), z

print('='*50)
print('5-SEED TEST')
print('='*50)

for seed in range(5):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = M()
    o = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(10):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            p, z = m(St[idx[i:i+32]], At[idx[i:i+32]].unsqueeze(-1))
            F.mse_loss(p, SNt[idx[i:i+32]]).backward()
            o.step(); o.zero_grad()
    
    with torch.no_grad():
        _, z = m(St, At.unsqueeze(-1))
    
    ni = T=='n'; ti = T=='t'
    r = (z[ni].std() - z[ti].std()) / z[ni].std() * 100
    print(f'seed {seed}: {r:+.1f}%')

print('='*50)
