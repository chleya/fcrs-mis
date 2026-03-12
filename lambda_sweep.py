"""
Lambda Sweep Experiment - Compression Pressure vs Variable Emergence

Tests whether variable emergence shows phase transition with compression coefficient λ

L = prediction + λ * compression

Run: python lambda_sweep.py
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
print('LAMBDA SWEEP - COMPRESSION PRESSURE')
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

# Model with compression
class CausalModel(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, latent_dim))
        self.dynamics = nn.Sequential(nn.Linear(latent_dim + 1, 24), nn.ReLU(), nn.Linear(24, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 24), nn.ReLU(), nn.Linear(24, 4))
    
    def forward(self, o, a):
        z = self.encoder(o)
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        return self.decoder(z_next), z

# Sweep lambda
lambdas = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

print('\n' + '='*60)
print('RESULTS')
print('='*60)
print(f"{'lambda':<10} | {'|Corr(v)|':>10} | {'|Corr(theta)|':>14} | {'Mean |Corr|':>12}")
print('-'*60)

results = []

for lam in lambdas:
    # Train
    model = CausalModel(latent_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(8):
        idx = np.random.permutation(len(S_t))
        for i in range(0, len(idx), 32):
            batch_idx = idx[i:i+32]
            o = S_t[batch_idx]
            a = A_t[batch_idx]
            s_next = S_next_t[batch_idx]
            
            pred, z = model(o, a)
            
            # Prediction loss
            pred_loss = F.mse_loss(pred, s_next)
            
            # Compression loss (L1 on latent)
            if lam > 0:
                comp_loss = lam * torch.mean(torch.abs(z))
            else:
                comp_loss = 0
            
            loss = pred_loss + comp_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        _, z = model(S_t, A_t)
    
    z_np = z.detach().numpy()
    
    # Correlation with velocity and angle
    corr_vel = abs(np.corrcoef(z_np[:, 0], np.array(states)[:, 1])[0, 1])
    corr_angle = abs(np.corrcoef(z_np[:, 1], np.array(states)[:, 2])[0, 1])
    mean_corr = np.mean([corr_vel, corr_angle])
    
    results.append({
        'lambda': lam,
        'corr_vel': corr_vel,
        'corr_angle': corr_angle,
        'mean_corr': mean_corr
    })
    
    print(f"{lam:<10} | {corr_vel:>10.3f} | {corr_angle:>14.3f} | {mean_corr:>12.3f}")

print('='*60)

# Analysis
print('\n' + '='*60)
print('PHASE TRANSITION ANALYSIS')
print('='*60)

corrs = [r['mean_corr'] for r in results]
lam_corr_pairs = [(r['lambda'], r['mean_corr']) for r in results]

# Find optimal lambda
best_idx = np.argmax(corrs)
best_lambda = results[best_idx]['lambda']
best_corr = results[best_idx]['mean_corr']

print(f'\nBest lambda: {best_lambda} with |Corr| = {best_corr:.3f}')

# Check for phase transition pattern
print('\nPattern analysis:')
print('-'*40)

# Check low lambda region
low_lam_corrs = [r['mean_corr'] for r in results if r['lambda'] < 0.001]
high_lam_corrs = [r['mean_corr'] for r in results if r['lambda'] > 0.01]

if len(low_lam_corrs) > 0 and len(high_lam_corrs) > 0:
    avg_low = np.mean(low_lam_corrs)
    avg_high = np.mean(high_lam_corrs)
    print(f'Low lambda (<0.001): avg |Corr| = {avg_low:.3f}')
    print(f'High lambda (>0.01): avg |Corr| = {avg_high:.3f}')
    
    if avg_high > avg_low + 0.1:
        print('')
        print('=> Phase transition: compression enhances variable emergence!')
    elif avg_high < avg_low - 0.1:
        print('')
        print('=> Inverse: high compression hurts variable emergence')
    else:
        print('')
        print('=> No clear phase transition')

# Visualization
print('\n' + '='*60)
print('VISUALIZATION')
print('='*60)
for r in results:
    bar_len = int(r['mean_corr'] * 20)
    print(f"lam={r['lambda']:<6} | {'#' * bar_len}{'-' * (20-bar_len)} | {r['mean_corr']:.3f}")

print('\n' + '='*60)
print('CONCLUSION')
print('='*60)
print('Testing compression pressure effect on variable emergence...')
