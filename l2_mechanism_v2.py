"""
L2 Mechanism v2 - Stronger Intervention + Composition Loss
Tests whether stronger intervention can trigger mechanism formation

Key improvements:
1. Multi-step intervention (fix latent over multiple steps)
2. Composition loss (encourage variable binding)
3. Larger intervention effect

Run: python l2_mechanism_v2.py
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
print('L2 MECHANISM v2 - STRONGER INTERVENTION')
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

env = CartPole()

# Generate data with STRONGER interventions
def generate_strong_intervention_data(env, n_samples=3000):
    """
    Strong interventions:
    1. Normal transitions
    2. HARD theta intervention (force to 0)
    3. HARD x intervention (force to 0)
    4. JOINT intervention (both theta and x fixed)
    """
    normal = []
    theta_int = []
    x_int = []
    joint_int = []
    
    for _ in range(n_samples):
        # Normal
        s = env.reset()
        for _ in range(15):
            a = np.random.randint(0, 2)
            s_next = env.step(s, a)
            normal.append({'s': s.copy(), 'a': a, 's_next': s_next.copy(), 'type': 'normal'})
            s = s_next
        
        # HARD theta intervention
        s = env.reset()
        for _ in range(10):
            a = np.random.randint(0, 2)
            s_next = env.step(s, a)
            # Force theta = 0
            s_next[2] = 0.0
            theta_int.append({'s': s.copy(), 'a': a, 's_next': s_next.copy(), 'type': 'theta_0'})
            s = s_next
        
        # HARD x intervention
        s = env.reset()
        for _ in range(10):
            a = np.random.randint(0, 2)
            s_next = env.step(s, a)
            # Force x = 0
            s_next[0] = 0.0
            x_int.append({'s': s.copy(), 'a': a, 's_next': s_next.copy(), 'type': 'x_0'})
            s = s_next
        
        # JOINT intervention
        s = env.reset()
        for _ in range(10):
            a = np.random.randint(0, 2)
            s_next = env.step(s, a)
            s_next[0] = 0.0
            s_next[2] = 0.0
            joint_int.append({'s': s.copy(), 'a': a, 's_next': s_next.copy(), 'type': 'joint'})
            s = s_next
    
    return normal, theta_int, x_int, joint_int

print('\nGenerating strong intervention data...')
normal, theta_int, x_int, joint_int = generate_strong_intervention_data(env, 3000)

all_data = normal + theta_int + x_int + joint_int
np.random.shuffle(all_data)

states = np.array([d['s'] for d in all_data])
actions = np.array([d['a'] for d in all_data])
next_states = np.array([d['s_next'] for d in all_data])

S_t = torch.FloatTensor(states)
A_t = torch.FloatTensor(actions).float().unsqueeze(-1)
S_next_t = torch.FloatTensor(next_states)

print(f'Total samples: {len(all_data)}')
print(f'Normal: {len(normal)}, Theta: {len(theta_int)}, X: {len(x_int)}, Joint: {len(joint_int)}')

# Model with composition loss
class CausalMechanismV2(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, latent_dim))
        
        # Independent dynamics per latent
        self.dynamics = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim + 1, 16), nn.ReLU(), nn.Linear(16, 1))
            for _ in range(latent_dim)
        ])
        
        # Composition predictor
        self.composition = nn.Sequential(
            nn.Linear(latent_dim * 2, 32), nn.ReLU(),
            nn.Linear(32, latent_dim))
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 4))
    
    def forward(self, o, a, intervene_idx=None, intervene_value=None):
        z = self.encoder(o)
        
        # Independent dynamics
        z_delta = []
        for i in range(self.latent_dim):
            delta_i = self.dynamics[i](torch.cat([z, a], dim=-1))
            z_delta.append(delta_i)
        
        z_delta = torch.cat(z_delta, dim=-1)
        z_next = z + z_delta
        
        # Composition
        z_compose = self.composition(torch.cat([z, z_next], dim=-1))
        
        # Intervention
        if intervene_idx is not None:
            z_next[:, intervene_idx] = intervene_value
        
        recon = self.decoder(z_next)
        
        return recon, z, z_compose

# Train
print('\nTraining...')
model = CausalMechanismV2(latent_dim=4)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(12):
    idx = np.random.permutation(len(S_t))
    for i in range(0, len(idx), 32):
        batch_idx = idx[i:i+32]
        o = S_t[batch_idx]
        a = A_t[batch_idx]
        s_next = S_next_t[batch_idx]
        
        pred, z, z_compose = model(o, a)
        
        # Reconstruction loss
        loss_rec = F.mse_loss(pred, s_next)
        
        # Composition loss (encourage binding)
        loss_comp = F.mse_loss(z_compose, z)
        
        loss = loss_rec + 0.2 * loss_comp
        
        opt.zero_grad()
        loss.backward()
        opt.step()

# Evaluate
model.eval()
with torch.no_grad():
    pred, z, z_compose = model(S_t, A_t)

z_np = z.numpy()

# Analysis
print('\n' + '='*60)
print('L2 V2 RESULTS')
print('='*60)

print('\nLatent-State Correlations:')
for i in range(4):
    for j, name in enumerate(['x', 'x_dot', 'theta', 'theta_dot']):
        corr = np.corrcoef(z_np[:, i], states[:, j])[0, 1]
        if not np.isnan(corr):
            print(f'  z[{i}] - {name}: {corr:+.3f}')

# Intervention analysis
print('\n' + '='*60)
print('INTERVENTION EFFECT')
print('='*60)

# Split by type
normal_idx = [i for i, d in enumerate(all_data) if d['type'] == 'normal']
theta_idx = [i for i, d in enumerate(all_data) if d['type'] == 'theta_0']
x_idx = [i for i, d in enumerate(all_data) if d['type'] == 'x_0']
joint_idx = [i for i, d in enumerate(all_data) if d['type'] == 'joint']

z_normal = z_np[normal_idx]
z_theta = z_np[theta_idx]
z_x = z_np[x_idx]
z_joint = z_np[joint_idx]

print(f'\nNormal:     mean={z_normal.mean():.3f}, std={z_normal.std():.3f}')
print(f'Theta=0:    mean={z_theta.mean():.3f}, std={z_theta.std():.3f}')
print(f'X=0:        mean={z_x.mean():.3f}, std={z_x.std():.3f}')
print(f'Joint:      mean={z_joint.mean():.3f}, std={z_joint.std():.3f}')

# Variance reduction
theta_reduction = (z_normal.std() - z_theta.std()) / z_normal.std() * 100
x_reduction = (z_normal.std() - z_x.std()) / z_normal.std() * 100
joint_reduction = (z_normal.std() - z_joint.std()) / z_normal.std() * 100

print(f'\nVariance Reduction:')
print(f'  Theta intervention: {theta_reduction:+.1f}%')
print(f'  X intervention:     {x_reduction:+.1f}%')
print(f'  Joint intervention: {joint_reduction:+.1f}%')

# Target check
print('\n' + '='*60)
print('CONCLUSION')
print('='*60)
if theta_reduction >= 30 or x_reduction >= 30:
    print('=> MECHANISM DETECTED! Variance reduction >= 30%')
elif theta_reduction >= 15 or x_reduction >= 15:
    print('=> PARTIAL mechanism effect (15-30%)')
else:
    print('=> Weak mechanism effect (<15%)')
    print('=> Need stronger intervention or different architecture')
