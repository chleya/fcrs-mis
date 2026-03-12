"""
Fix B: Multi-seed capacity sweep
Averages over 5 seeds for robust results
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

print('='*60)
print('MULTI-SEED CAPACITY SWEEP')
print('='*60)

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

# Generate fixed data (same for all seeds)
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

print(f'Data: {len(states)} samples')

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

# Sweep with multiple seeds
dims = [2, 4, 8, 16, 32]
num_seeds = 5

print('\n' + '='*60)
print('RESULTS (5 seeds averaged)')
print('='*60)
print(f"{'dim':<6} | {'Mean |Corr|':>12} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
print('-'*60)

all_results = {}

for dim in dims:
    corrs = []
    
    for seed in range(num_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        model = CausalModel(dim)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        for epoch in range(8):
            idx = np.random.permutation(len(S_t))
            for i in range(0, len(idx), 32):
                batch_idx = idx[i:i+32]
                o = S_t[batch_idx]
                a = A_t[batch_idx]
                s_next = S_next_t[batch_idx]
                
                pred, z = model(o, a)
                loss = F.mse_loss(pred, s_next)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        model.eval()
        with torch.no_grad():
            _, z = model(S_t, A_t)
        
        z_np = z[:, 0].detach().numpy()
        corr = abs(np.corrcoef(z_np, np.array(states)[:, 1])[0, 1])
        corrs.append(corr)
    
    mean_corr = np.mean(corrs)
    std_corr = np.std(corrs)
    
    all_results[dim] = {
        'mean': mean_corr,
        'std': std_corr,
        'min': min(corrs),
        'max': max(corrs),
        'raw': corrs
    }
    
    print(f"{dim:<6} | {mean_corr:>12.3f} | {std_corr:>8.3f} | {min(corrs):>8.3f} | {max(corrs):>8.3f}")

print('='*60)

# Analysis
print('\nANALYSIS:')
print('-'*40)
dims_means = [all_results[d]['mean'] for d in dims]
dims_stds = [all_results[d]['std'] for d in dims]

# Find if there's a trend
max_dim = dims[np.argmax(dims_means)]
min_dim = dims[np.argmin(dims_means)]

print(f'Best dim: {max_dim} (mean={all_results[max_dim]["mean"]:.3f})')
print(f'Worst dim: {min_dim} (mean={all_results[min_dim]["mean"]:.3f})')

# Check if variance is high
avg_std = np.mean(dims_stds)
print(f'Average std: {avg_std:.3f}')

if avg_std < 0.1:
    print('\n=> Low variance - results are reliable')
else:
    print('\n=> High variance - need more seeds')

# Visualization
print('\nVISUALIZATION (error bars):')
for dim in dims:
    m = all_results[dim]['mean']
    s = all_results[dim]['std']
    bar_len = int(m * 20)
    print(f"dim={dim:>2} | {'#' * bar_len}{'-' * (20-bar_len)} | {m:.3f} +/- {s:.3f}")
