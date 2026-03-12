"""Quick multi-seed: 3 seeds per capacity"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def pend(s, a):
    t, o = s
    to = (a - 0.5) * 2
    on = o + (-9.8*np.sin(t) - 0.1*o + to) * 0.05
    tn = t + on * 0.05
    return np.array([tn, on])

class M(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, 24), nn.ReLU(), nn.Linear(24, dim))
        self.dyn = nn.Sequential(nn.Linear(dim+1, 24), nn.ReLU(), nn.Linear(24, dim))
        self.dec = nn.Sequential(nn.Linear(dim, 24), nn.ReLU(), nn.Linear(24, 2))
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

def test(dim, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    data = []
    for _ in range(1000):
        s = np.random.uniform(-0.5, 0.5, 2)
        for _ in range(10):
            a = np.random.randint(0, 2)
            sn = pend(s, a)
            data.append((s.copy(), a, sn.copy(), 'n'))
            s = sn
    
    for _ in range(200):
        s = np.random.uniform(-0.5, 0.5, 2); s[0] = 0.0
        for _ in range(8):
            a = np.random.randint(0, 2)
            sn = pend(s, a); sn[0] = 0.0
            data.append((s.copy(), a, sn.copy(), 'i'))
            s = sn
    
    S = np.array([d[0] for d in data])
    A = np.array([d[1] for d in data])
    SN = np.array([d[2] for d in data])
    T = np.array([d[3] for d in data])
    
    St = torch.FloatTensor(S)
    At = torch.FloatTensor(A).float().unsqueeze(-1)
    SNt = torch.FloatTensor(SN)
    
    m = M(dim)
    o = torch.optim.Adam(m.parameters(), lr=3e-4)
    for _ in range(6):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            p, z = m(St[idx[i:i+32]], At[idx[i:i+32]])
            loss = F.mse_loss(p, SNt[idx[i:i+32]])
            loss.backward()
            o.step()
            o.zero_grad()
    
    m.eval()
    with torch.no_grad():
        p, z = m(St, At)
    
    z_np = z.numpy()
    corrs = [abs(np.corrcoef(z_np[:,i], S[:,1])[0,1]) for i in range(min(dim,2)) if i < z_np.shape[1]]
    corr = max(corrs) if corrs else 0
    
    ni, ii = T=='n', T=='i'
    ratio = np.mean((p.numpy()[ii]-SN[ii])**2) / np.mean((p.numpy()[ni]-SN[ni])**2)
    
    return corr, ratio

dims = [2, 3, 4, 6, 8]
seeds = [0, 1, 2]

print('MULTI-SEED (3 seeds)')
print('dim | corr_mean+/-std | ratio_mean+/-std')
print('-'*50)

for dim in dims:
    cs, rs = [], []
    for s in seeds:
        c, r = test(dim, s)
        cs.append(c)
        rs.append(r)
    print(f'{dim:>3} | {np.mean(cs):.3f}+/-{np.std(cs):.3f} | {np.mean(rs):.3f}+/-{np.std(rs):.3f}')
