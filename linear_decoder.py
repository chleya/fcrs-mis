"""Linear decoder test: Wz -> theta, omega"""
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

# Train causal model with dim=2
m = M(2)
o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(10):
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

# Linear decoder: Wz -> true variables
print('\n' + '='*50)
print('LINEAR DECODER TEST')
print('='*50)

# Simple linear regression
from numpy.linalg import lstsq

# Decode theta
W_theta, _, _, _ = lstsq(z_np, S[:, 0], rcond=None)
theta_pred = z_np @ W_theta
r2_theta = 1 - np.sum((theta_pred - S[:, 0])**2) / np.sum((S[:, 0] - np.mean(S[:, 0]))**2)

# Decode omega
W_omega, _, _, _ = lstsq(z_np, S[:, 1], rcond=None)
omega_pred = z_np @ W_omega
r2_omega = 1 - np.sum((omega_pred - S[:, 1])**2) / np.sum((S[:, 1] - np.mean(S[:, 1]))**2)

print(f'\nR^2 for theta: {r2_theta:.4f}')
print(f'R^2 for omega: {r2_omega:.4f}')
print(f'Average R^2: {(r2_theta + r2_omega)/2:.4f}')

print('\n' + '='*50)
print('DIRECT CORRELATION')
print('='*50)
for i in range(2):
    c_theta = np.corrcoef(z_np[:, i], S[:, 0])[0, 1]
    c_omega = np.corrcoef(z_np[:, i], S[:, 1])[0, 1]
    print(f'z[{i}] vs theta: {c_theta:+.3f}')
    print(f'z[{i}] vs omega: {c_omega:+.3f}')

print('\n' + '='*50)
print('INTERPRETATION')
print('='*50)
print('If R^2 > 0.95: variable subspace recovered!')
print(f'Current: R^2 = {(r2_theta + r2_omega)/2:.3f}')
