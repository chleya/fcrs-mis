"""L2 v2 simplified"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# CartPole
def step(s, a):
    x, xd, t, td = s
    f = 10.0 if a==1 else -10.0
    ct, st = np.cos(t), np.sin(t)
    tmp = f/1.1 + 0.05*td**2*st
    ta = (9.8*st - ct*tmp)/(0.5*(4/3-0.1*ct**2/1.1))
    xa = tmp - 0.05*ta*ct/1.1
    return np.array([x+0.02*xd, xd+0.02*xa, t+0.02*td, td+0.02*ta])

# Data
data = []
for _ in range(2000):
    s = np.random.uniform(-0.05, 0.05, 4)
    for _ in range(20):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        data.append((s.copy(), a, sn.copy()))
        s = sn

S = np.array([d[0] for d in data])
A = np.array([d[1] for d in data])
SN = np.array([d[2] for d in data])

# Normal vs intervention
normal_idx = list(range(len(data)//2))
int_idx = list(range(len(data)//2, len(data)))

St = torch.FloatTensor(S)
At = torch.FloatTensor(A).float().unsqueeze(-1)
SNt = torch.FloatTensor(SN)

# Model
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], dim=-1))
        return self.dec(zn), z

m = M()
o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(10):
    idx = np.random.permutation(len(St))
    for i in range(0, len(idx), 32):
        p, z = m(St[idx[i:i+32]], At[idx[i:i+32]])
        loss = F.mse_loss(p, SNt[idx[i:i+32]])
        loss.backward(); o.step(); o.zero_grad()

m.eval()
with torch.no_grad():
    _, z = m(St, At)

z_np = z.numpy()

# Results
print('='*50)
print('L2 V2 SIMPLE RESULTS')
print('='*50)
print('\nCorrelations:')
for i in range(4):
    for j, n in enumerate(['x','xd','t','td']):
        c = np.corrcoef(z_np[:,i], S[:,j])[0,1]
        if not np.isnan(c):
            print(f'z[{i}]-{n}: {c:+.3f}')

# Variance
z_n = z_np[normal_idx]
z_i = z_np[int_idx]
print(f'\nNormal std: {z_n.std():.3f}')
print(f'Intervention std: {z_i.std():.3f}')
print(f'Reduction: {(z_n.std()-z_i.std())/z_n.std()*100:+.1f}%')
