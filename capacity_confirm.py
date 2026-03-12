"""Capacity confirmation test - dim=6,10,12"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CartPole
class CartPole:
    def __init__(self):
        self.force_mag = 10.0
        self.tau = 0.02
        
    def step(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = force / 1.1 + 0.05 * theta_dot**2 * sintheta
        thetaacc = (9.8*sintheta - costheta*temp) / (0.5 * (4/3 - 0.1*costheta**2/1.1))
        xacc = temp - 0.05*thetaacc*costheta/1.1
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return np.array([x, x_dot, theta, theta_dot])
    
    def reset(self):
        return np.random.uniform(-0.05, 0.05, size=4)

env = CartPole()
states, actions, next_states = [], [], []
for _ in range(200):
    s = env.reset()
    for _ in range(30):
        a = np.random.randint(0, 2)
        s_next = env.step(s, a)
        states.append(s.copy())
        actions.append(a)
        next_states.append(s_next.copy())
        s = s_next

S_t = torch.FloatTensor(np.array(states))
A_t = torch.FloatTensor(np.array(actions)).float().unsqueeze(-1)
S_next_t = torch.FloatTensor(np.array(next_states))

class CausalModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, latent_dim))
        self.dynamics = nn.Sequential(nn.Linear(latent_dim + 1, 24), nn.ReLU(), nn.Linear(24, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 24), nn.ReLU(), nn.Linear(24, 4))
    
    def forward(self, o, a):
        z = self.encoder(o)
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        return self.decoder(z_next), z

print('CAPACITY CONFIRMATION')
print('='*50)
dims = [2, 4, 6, 8, 10, 12, 16, 32]
results = []

for dim in dims:
    model = CausalModel(dim)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for _ in range(8):
        idx = np.random.permutation(len(S_t))
        for i in range(0, len(idx), 32):
            pred, z = model(S_t[idx[i:i+32]], A_t[idx[i:i+32]])
            loss = F.mse_loss(pred, S_next_t[idx[i:i+32]])
            loss.backward()
            opt.step()
            opt.zero_grad()
    
    model.eval()
    with torch.no_grad():
        _, z = model(S_t, A_t)
    
    z_np = z[:, 0].numpy()
    corr = np.abs(np.corrcoef(z_np, np.array(states)[:, 1])[0, 1])
    results.append((dim, corr))
    print(f'dim={dim:>2} | Corr={corr:.3f}')

print()
print('ANALYSIS:')
print('-'*50)
# Check if dim=8 is isolated
vals = dict(results)
if vals[8] < 0.3:
    if vals[6] > 0.7 and vals[10] > 0.7:
        print('dim=8 is ISOLATED instability point')
    else:
        print('dim=8 is part of gradual decline')
