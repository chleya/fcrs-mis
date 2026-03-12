"""
L2 Mechanism Emergence Experiment
Tests whether intervention on one variable affects another - core of causal mechanisms

Key question: If we intervene on variable A, does variable B change?
This is the essence of causal mechanisms.

Run: python l2_mechanism_experiment.py
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
print('L2 CAUSAL MECHANISM EMERGENCE')
print('='*60)

# CartPole - has clear causal mechanisms
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

# Generate data with interventions
def generate_intervention_data(env, n_samples=2000):
    """
    Generate data with known interventions
    - Normal transitions
    - Intervene on theta (force angle change)
    - Intervene on x (force position change)
    """
    normal_data = []
    theta_intervention = []
    x_intervention = []
    
    for _ in range(n_samples):
        s = env.reset()
        
        # Normal transition
        for _ in range(10):
            a = np.random.randint(0, 2)
            s_next = env.step(s, a)
            normal_data.append({
                'state': s.copy(),
                'action': a,
                'next_state': s_next.copy(),
                'type': 'normal'
            })
            s = s_next
        
        # Intervene on theta
        s = env.reset()
        theta_fixed = np.random.uniform(-0.2, 0.2)
        for _ in range(5):
            a = np.random.randint(0, 2)
            s_next = env.step(s, a)
            # Override theta
            s_next[2] = theta_fixed
            theta_intervention.append({
                'state': s.copy(),
                'action': a,
                'next_state': s_next.copy(),
                'type': 'theta_fixed'
            })
            s = s_next
        
        # Intervene on x
        s = env.reset()
        x_fixed = 0.0
        for _ in range(5):
            a = np.random.randint(0, 2)
            s_next = env.step(s, a)
            # Override x
            s_next[0] = x_fixed
            x_intervention.append({
                'state': s.copy(),
                'action': a,
                'next_state': s_next.copy(),
                'type': 'x_fixed'
            })
            s = s_next
    
    return normal_data, theta_intervention, x_intervention

env = CartPole()
normal, theta_int, x_int = generate_intervention_data(env, 2000)

print(f'Normal: {len(normal)}')
print(f'Theta intervention: {len(theta_int)}')
print(f'X intervention: {len(x_int)}')

# Prepare data
all_data = normal + theta_int + x_int
np.random.shuffle(all_data)

states = np.array([d['state'] for d in all_data])
actions = np.array([d['action'] for d in all_data])
next_states = np.array([d['next_state'] for d in all_data])

S_t = torch.FloatTensor(states)
A_t = torch.FloatTensor(actions).float().unsqueeze(-1)
S_next_t = torch.FloatTensor(next_states)

# Model: Causal with mechanism learning
class CausalMechanism(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: state -> latent
        self.encoder = nn.Sequential(
            nn.Linear(4, 24), nn.ReLU(),
            nn.Linear(24, latent_dim))
        
        # Dynamics: [z, action] -> delta_z
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 1, 24), nn.ReLU(),
            nn.Linear(24, latent_dim))
        
        # Mechanism predictor: predict next latent from current
        self.mechanism = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.ReLU(),
            nn.Linear(24, latent_dim))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.ReLU(),
            nn.Linear(24, 4))
    
    def forward(self, o, a):
        z = self.encoder(o)
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        
        # Mechanism: predict next latent without action
        z_next_mech = self.mechanism(z)
        
        return self.decoder(z_next), self.decoder(z_next_mech), z

# Train
print('\nTraining...')
model = CausalMechanism(latent_dim=4)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(10):
    idx = np.random.permutation(len(S_t))
    for i in range(0, len(idx), 32):
        batch_idx = idx[i:i+32]
        o = S_t[batch_idx]
        a = A_t[batch_idx]
        s_next = S_next_t[batch_idx]
        
        pred, pred_mech, z = model(o, a)
        
        # Two losses
        loss_action = F.mse_loss(pred, s_next)
        loss_mechanism = F.mse_loss(pred_mech, s_next)
        
        # Total loss
        loss = loss_action + 0.1 * loss_mechanism
        
        opt.zero_grad()
        loss.backward()
        opt.step()

# Evaluate
model.eval()
with torch.no_grad():
    pred, pred_mech, z = model(S_t, A_t)

z_np = z.numpy()

# Analyze mechanism emergence
print('\n' + '='*60)
print('MECHANISM ANALYSIS')
print('='*60)

# Check correlation between latents and true states
print('\nLatent-State Correlations:')
for i in range(4):
    for j, name in enumerate(['x', 'x_dot', 'theta', 'theta_dot']):
        corr = np.corrcoef(z_np[:, i], states[:, j])[0, 1]
        if not np.isnan(corr):
            print(f'  z[{i}] - {name}: {corr:+.3f}')

# Check if intervention affects latent dynamics
print('\nIntervention Effect on Latents:')

# Split by intervention type
normal_idx = [i for i, d in enumerate(all_data) if d['type'] == 'normal']
theta_idx = [i for i, d in enumerate(all_data) if d['type'] == 'theta_fixed']
x_idx = [i for i, d in enumerate(all_data) if d['type'] == 'x_fixed']

z_normal = z_np[normal_idx]
z_theta = z_np[theta_idx]
z_x = z_np[x_idx]

print(f'\nNormal transitions: z mean = {z_normal.mean():.3f}, std = {z_normal.std():.3f}')
print(f'Theta fixed: z mean = {z_theta.mean():.3f}, std = {z_theta.std():.3f}')
print(f'X fixed: z mean = {z_x.mean():.3f}, std = {z_x.std():.3f}')

# Check if intervention changes latent dynamics
print('\n' + '='*60)
print('CONCLUSION')
print('='*60)
print('Testing whether interventions affect learned latents...')

if z_theta.std() < z_normal.std() * 0.5:
    print('=> Theta intervention reduces latent variance - mechanism detected!')
else:
    print('=> Latents not strongly affected by intervention')

if z_x.std() < z_normal.std() * 0.5:
    print('=> X intervention reduces latent variance - mechanism detected!')
else:
    print('=> Latents not strongly affected by X intervention')
