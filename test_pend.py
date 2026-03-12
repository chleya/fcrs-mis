"""Pendulum test"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
random.seed(42); np.random.seed(42); torch.manual_seed(42)

def pend(s, a):
    t, o = s
    to = (a-.5)*2
    on = o + (-9.8*np.sin(t)-.1*o+to)*.05
    tn = t + on*.05
    return np.array([tn, on])

class M(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.enc = nn.Sequential(nn.Linear(dim, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, dim))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z,a],-1))
        return self.dec(zn), z

# Generate data
d = []
for _ in range(1000):
    s = np.random.uniform(-.5, .5, 2)
    for _ in range(10):
        a = np.random.randint(0, 2)
        sn = pend(s, a)
        d.append((s.copy(), a, sn.copy(), 'n')); s = sn
for _ in range(300):
    s = np.random.uniform(-.5, .5, 2); s[0]=0
    for _ in range(8):
        a = np.random.randint(0, 2)
        sn = pend(s, a); sn[0]=0
        d.append((s.copy(), a, sn.copy(), 'i')); s = sn

S = np.array([x[0] for x in d]); A = np.array([x[1] for x in d]); SN = np.array([x[2] for x in d]); T = np.array([x[3] for x in d])
St, At, SNt = map(torch.FloatTensor, [S, A, SN])

m = M(2)
o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(8):
    idx = np.random.permutation(len(St))
    for i in range(0, len(idx), 32):
        p, z = m(St[idx[i:i+32]], At[idx[i:i+32]].unsqueeze(-1))
        F.mse_loss(p, SNt[idx[i:i+32]]).backward(); o.step(); o.zero_grad()

ni, ii = T=='n', T=='i'
with torch.no_grad():
    p, z = m(St, At.unsqueeze(-1))
r = np.mean((p.numpy()[ii]-SN[ii])**2)/np.mean((p.numpy()[ni]-SN[ni])**2)
c = np.corrcoef(z.numpy()[:,0], S[:,0])[0,1]

print('C. PENDULUM:')
print(f'  ratio={r:.3f}')
print(f'  corr(z,theta)={c:.3f}')
