"""Quick A+B+C test"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
random.seed(42); np.random.seed(42); torch.manual_seed(42)

def step(s, a):
    x, xd, t, td = s
    f = 10.0 if a==1 else -10.0
    ct, st = np.cos(t), np.sin(t)
    tmp = f/1.1 + 0.05*td**2*st
    ta = (9.8*st - ct*tmp)/(0.5*(4/3-0.1*ct**2/1.1))
    return np.array([x+0.02*xd, xd+0.02*(tmp-0.05*ta*ct/1.1), t+0.02*td, td+0.02*ta])

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z,a],-1))
        return self.dec(zn), z

print('A. MULTI-SEED (CartPole x_fix):')
for sd in range(3):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    d, di = [], []
    for _ in range(1000):
        s = np.random.uniform(-0.05, 0.05, 4)
        for _ in range(10):
            a = np.random.randint(0, 2)
            sn = step(s, a)
            d.append((s.copy(), a, sn.copy(), 'n'))
            s = sn
    for _ in range(300):
        s = np.random.uniform(-0.05, 0.05, 4); s[0]=0
        for _ in range(8):
            a = np.random.randint(0, 2)
            sn = step(s, a); sn[0]=0
            di.append((s.copy(), a, sn.copy(), 'i')); s = sn
    D = d+di
    S = np.array([x[0] for x in D]); A = np.array([x[1] for x in D]); SN = np.array([x[2] for x in D]); T = np.array([x[3] for x in D])
    St, At, SNt = map(torch.FloatTensor, [S, A, SN])
    m = M(); o = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(8):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            p, z = m(St[idx[i:i+32]], At[idx[i:i+32]].unsqueeze(-1))
            F.mse_loss(p, SNt[idx[i:i+32]]).backward(); o.step(); o.zero_grad()
    ni, ii = T=='n', T=='i'
    with torch.no_grad():
        p, z = m(St, At.unsqueeze(-1))
    r = np.mean((p.numpy()[ii]-SN[ii])**2)/np.mean((p.numpy()[ni]-SN[ni])**2)
    c = np.corrcoef(z.numpy()[:,0], S[:,1])[0,1]
    print(f'  seed{sd}: ratio={r:.3f}, corr={c:.3f}')

print('B. TRAJECTORY:')
d = []
s = np.array([0,0,0.1,0])
for _ in range(20):
    a = 1
    sn = step(s, a)
    d.append((s.copy(), a, sn.copy())); s = sn
S = np.array([x[0] for x in d]); A = np.array([x[1] for x in d]); SN = np.array([x[2] for x in d])
St, At, SNt = map(torch.FloatTensor, [S, A, SN])
m = M(); o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(10):
    for i in range(0, len(St), 8):
        p, z = m(St[i:i+8], At[i:i+8].unsqueeze(-1))
        F.mse_loss(p, SNt[i:i+8]).backward(); o.step(); o.zero_grad()
with torch.no_grad():
    p, z = m(St, At.unsqueeze(-1))
print('Step | True_x | Pred_x | Error')
for i in range(10):
    print(f'{i:4} | {SN[i,0]:7.3f} | {p.numpy()[i,0]:7.3f} | {abs(p.numpy()[i,0]-SN[i,0]):7.3f}')

print('C. PENDULUM (quick):')
def pend(s, a):
    t, o = s
    to = (a-.5)*2
    on = o + (-9.8*np.sin(t)-.1*o+to)*.05
    tn = t + on*.05
    return np.array([tn, on])
d = []
for _ in range(800):
    s = np.random.uniform(-.5, .5, 2)
    for _ in range(10):
        a = np.random.randint(0, 2)
        sn = pend(s, a)
        d.append((s.copy(), a, sn.copy(), 'n')); s = sn
for _ in range(200):
    s = np.random.uniform(-.5, .5, 2); s[0]=0
    for _ in range(8):
        a = np.random.randint(0, 2)
        sn = pend(s, a); sn[0]=0
        d.append((s.copy(), a, sn.copy(), 'i')); s = sn
S = np.array([x[0] for x in d]); A = np.array([x[1] for x in d]); SN = np.array([x[2] for x in d]); T = np.array([x[3] for x in d])
St, At, SNt = map(torch.FloatTensor, [S, A, SN])
m = M(); o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(8):
    idx = np.random.permutation(len(St))
    for i in range(0, len(idx), 32):
        p, z = m(St[idx[i:i+32]], At[idx[i:i+32]].unsqueeze(-1))
        F.mse_loss(p, SNt[idx[i:i+32]]).backward(); o.step(); o.zero_grad()
ni, ii = T=='n', T=='i'
with torch.no_grad():
    p, z = m(St, At.unsqueeze(-1))
r = np.mean((p.numpy()[ii]-SN[ii])**2)/np.mean((p.numpy()[ni]-SN[ni])**2)
print(f'  ratio={r:.3f}')
