"""
Capacity Phase Transition Experiment - Day 5
Tests whether variable emergence shows a phase transition with latent capacity

Sweep: latent_dim = 2, 4, 8, 16, 32
Also sweep λ (compression coefficient) to create phase diagram

Run: python capacity_sweep.py
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

print('='*60)
print('CAPACITY PHASE TRANSITION EXPERIMENT')
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

# Generate data
def generate_data(env, n_episodes=200, max_steps=30):
    states, actions, next_states = [], [], []
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = np.random.randint(0, 2)
            s_next = env.step(s, a)
            states.append(s.copy())
            actions.append(a)
            next_states.append(s_next.copy())
            s = s_next
    return np.array(states), np.array(actions), np.array(next_states)

env = CartPole()
states, actions, next_states = generate_data(env)

S_t = torch.FloatTensor(states)
A_t = torch.FloatTensor(actions).float().unsqueeze(-1)
S_next_t = torch.FloatTensor(next_states)

print(f'Data: {len(states)} samples')

# Causal model with variable latent_dim
class CausalModel(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, latent_dim))
        self.dynamics = nn.Sequential(nn.Linear(latent_dim + 1, 24), nn.ReLU(), nn.Linear(24, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 24), nn.ReLU(), nn.Linear(24, 4))
    
    def forward(self, o, a):
        z = self.encoder(o)
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        return self.decoder(z_next), z

# Sweep latent_dim
latent_dims = [2, 4, 8, 16, 32]

print('\n' + '='*60)
print('LATENT DIMENSION SWEEP')
print('='*60)
print(f"{'latent_dim':<12} | {'|Corr(v)|':>10} | {'|Corr(θ)|':>10} | {'Mean |Corr|':>12}")
print('-'*60)

results = []
for latent_dim in latent_dims:
    # Train
    model = CausalModel(latent_dim)
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
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        _, z = model(S_t, A_t)
    
    z_np = z[:, 0].detach().numpy()
    
    corr_vel = abs(np.corrcoef(z_np, states[:, 1])[0, 1])
    corr_angle = abs(np.corrcoef(z_np, states[:, 2])[0, 1])
    mean_corr = np.mean([corr_vel, corr_angle])
    
    results.append({
        'latent_dim': latent_dim,
        'corr_vel': corr_vel,
        'corr_angle': corr_angle,
        'mean_corr': mean_corr
    })
    
    print(f"{latent_dim:<12} | {corr_vel:>10.3f} | {corr_angle:>10.3f} | {mean_corr:>12.3f}")

print('='*60)

# Analysis
print('\n' + '='*60)
print('PHASE TRANSITION ANALYSIS')
print('='*60)

# Find phase transition
mean_corrs = [r['mean_corr'] for r in results]
max_corr = max(mean_corrs)
min_corr = min(mean_corrs)

print(f'\nMin |Corr|: {min_corr:.3f} (latent_dim={results[np.argmin(mean_corrs)]["latent_dim"]})')
print(f'Max |Corr|: {max_corr:.3f} (latent_dim={results[np.argmax(mean_corrs)]["latent_dim"]})')

# Check if there's a phase transition
corr_diffs = [mean_corrs[i+1] - mean_corrs[i] for i in range(len(mean_corrs)-1)]
max_diff_idx = np.argmax(corr_diffs)

if abs(corr_diffs[max_diff_idx]) > 0.1:
    print('')
    print('Phase transition detected!')
    print(f'  Transition around latent_dim = {latent_dims[max_diff_idx]} -> {latent_dims[max_diff_idx+1]}')
    print(f'  Change: {corr_diffs[max_diff_idx]:+.3f}')
else:
    print('')
    print('No sharp phase transition observed')
    print(f'  Gradual change: {corr_diffs}')

# Visual
print('\n' + '='*60)
print('VISUALIZATION')
print('='*60)
for r in results:
    bar_len = int(r['mean_corr'] * 20)
    print(f"dim={r['latent_dim']:>2} | {'#' * bar_len}{'-' * (20-bar_len)} | {r['mean_corr']:.3f}")

print('\n' + '='*60)
print('CONCLUSION')
print('='*60)
print('Testing whether variable emergence shows capacity-dependent phase transition...')
