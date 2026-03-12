"""
L2 Mechanism Generalization: Test random_noise intervention on Pendulum
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Pendulum
def pendulum_step(s, a):
    theta, omega = s
    torque = (a - 0.5) * 2
    omega_next = omega + (-9.8/1.0 * np.sin(theta) - 0.1 * omega + torque) * 0.05
    theta_next = theta + omega_next * 0.05
    theta_next = ((theta_next + np.pi) % (2*np.pi)) - np.pi
    return np.array([theta_next, omega_next])

# CartPole
def cartpole_step(s, a):
    x, xd, t, td = s
    f = 10.0 if a==1 else -10.0
    ct, st = np.cos(t), np.sin(t)
    tmp = f/1.1 + 0.05*td**2*st
    ta = (9.8*st - ct*tmp)/(0.5*(4/3-0.1*ct**2/1.1))
    return np.array([x+0.02*xd, xd+0.02*(tmp-0.05*ta*ct/1.1), t+0.02*td, td+0.02*ta])

def generate_data(env_name, n_normal=1500, n_noise=400):
    if env_name == 'pendulum':
        step_fn = pendulum_step
        dim = 2
    else:
        step_fn = cartpole_step
        dim = 4
    
    data = []
    
    # Normal
    for _ in range(n_normal):
        if env_name == 'pendulum':
            s = np.random.uniform(-0.5, 0.5, 2)
        else:
            s = np.random.uniform(-0.05, 0.05, 4)
        
        for _ in range(15):
            a = np.random.randint(0, 2)
            sn = step_fn(s, a)
            data.append((s.copy(), a, sn.copy(), 'normal'))
            s = sn
    
    # Noise intervention
    for _ in range(n_noise):
        if env_name == 'pendulum':
            s = np.random.uniform(-0.5, 0.5, 2)
        else:
            s = np.random.uniform(-0.05, 0.05, 4)
        
        for _ in range(10):
            a = np.random.randint(0, 2)
            sn = step_fn(s, a)
            sn += np.random.randn(dim) * 0.5  # Noise
            data.append((s.copy(), a, sn.copy(), 'noise'))
            s = sn
    
    return data

class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.enc = nn.Sequential(nn.Linear(dim, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, dim))
    
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

print('='*60)
print('GENERALIZATION TEST: Pendulum + Noise Intervention')
print('='*60)

# Pendulum
print('\nGenerating Pendulum data...')
p_data = generate_data('pendulum')
S = np.array([d[0] for d in p_data])
A = np.array([d[1] for d in p_data])
SN = np.array([d[2] for d in p_data])
T = np.array([d[3] for d in p_data])

St = torch.FloatTensor(S)
At = torch.FloatTensor(A).float().unsqueeze(-1)
SNt = torch.FloatTensor(SN)

print(f'Pendulum samples: {len(p_data)}')

# Train
m = Model(2)
o = torch.optim.Adam(m.parameters(), lr=3e-4)
for _ in range(10):
    idx = np.random.permutation(len(St))
    for i in range(0, len(idx), 32):
        p, z = m(St[idx[i:i+32]], At[idx[i:i+32]])
        F.mse_loss(p, SNt[idx[i:i+32]]).backward()
        o.step(); o.zero_grad()

m.eval()
with torch.no_grad():
    _, z = m(St, At)

z_np = z.numpy()

# Results
ni = T=='normal'
ii = T=='noise'

print('\n' + '='*60)
print('PENDULUM RESULTS')
print('='*60)

print('\nVariance:')
print(f'  Normal: {z_np[ni].std():.3f}')
print(f'  Noise:  {z_np[ii].std():.3f}')

variance_change = (z_np[ii].std() - z_np[ni].std()) / z_np[ni].std() * 100
print(f'  Change: {variance_change:+.1f}%')

print('\nCorrelations:')
for i in range(4):
    for j, name in enumerate(['theta', 'omega'][:2]):
        c = np.corrcoef(z_np[:, i], S[:, j])[0, 1]
        if not np.isnan(c):
            print(f'  z[{i}]-{name}: {c:+.3f}')

# Summary
print('\n' + '='*60)
print('COMPARISON')
print('='*60)
print(f'CartPole (previous):  -117%')
print(f'Pendulum (current):   {variance_change:+.1f}%')

if variance_change < -30:
    print('\n=> MECHANISM TRANSFERRED!')
elif variance_change < 0:
    print('\n=> Partial mechanism')
else:
    print('\n=> No mechanism in Pendulum')
