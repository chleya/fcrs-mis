"""
Fix A: Lambda with KL-divergence compression
Uses information bottleneck style: L = prediction - β * KL(q(z|x) || p(z))
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
print('KL-DIVERGENCE COMPRESSION TEST')
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

# VAE-style model with KL divergence
class VAECausalModel(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder produces mean and logvar
        self.encoder_mu = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, latent_dim))
        self.encoder_logvar = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, latent_dim))
        
        # Dynamics
        self.dynamics = nn.Sequential(nn.Linear(latent_dim + 1, 24), nn.ReLU(), nn.Linear(24, latent_dim))
        
        # Decoder
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 24), nn.ReLU(), nn.Linear(24, 4))
    
    def forward(self, o, a):
        mu = self.encoder_mu(o)
        logvar = self.encoder_logvar(o)
        
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Dynamics
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        
        # Decode
        pred = self.decoder(z_next)
        
        return pred, z, kl

# Sweep beta
betas = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

print('\n' + '='*60)
print('KL DIVERGENCE SWEEP (beta = compression strength)')
print('='*60)
print(f"{'beta':<8} | {'|Corr(v)|':>10} | {'Mean |Corr|':>12}")
print('-'*60)

results = []

for beta in betas:
    model = VAECausalModel(latent_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(8):
        idx = np.random.permutation(len(S_t))
        for i in range(0, len(idx), 32):
            batch_idx = idx[i:i+32]
            o = S_t[batch_idx]
            a = A_t[batch_idx]
            s_next = S_next_t[batch_idx]
            
            pred, z, kl = model(o, a)
            
            # Loss = prediction - beta * KL
            pred_loss = F.mse_loss(pred, s_next)
            loss = pred_loss - beta * kl / (32 * 4)  # normalize KL
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        _, z, _ = model(S_t, A_t)
    
    z_np = z.detach().numpy()
    corr_vel = abs(np.corrcoef(z_np[:, 0], np.array(states)[:, 1])[0, 1])
    corr_angle = abs(np.corrcoef(z_np[:, 1], np.array(states)[:, 2])[0, 1])
    mean_corr = np.mean([corr_vel, corr_angle])
    
    results.append({'beta': beta, 'corr_vel': corr_vel, 'corr_angle': corr_angle, 'mean_corr': mean_corr})
    print(f"{beta:<8} | {corr_vel:>10.3f} | {mean_corr:>12.3f}")

print('='*60)

# Find best
best = max(results, key=lambda x: x['mean_corr'])
print(f'\nBest beta: {best["beta"]} with |Corr| = {best["mean_corr"]:.3f}')

# Visualization
print('\nVISUALIZATION:')
for r in results:
    bar_len = int(r['mean_corr'] * 20)
    print(f"beta={r['beta']:<5} | {'#' * bar_len}{'-' * (20-bar_len)} | {r['mean_corr']:.3f}")
