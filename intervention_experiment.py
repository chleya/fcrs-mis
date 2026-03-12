"""
Intervention Experiment - Day 4
Tests whether intervention (active perturbation) improves variable emergence

Three conditions:
1. Baseline: observe → predict
2. Random Intervention: random action perturbation
3. Structured Intervention: targeted action (e.g., always accelerate)

Also tests action freezing: a = 0 (no control signal)

Run: python intervention_experiment.py
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
print('INTERVENTION EXPERIMENT - DAY 4')
print('='*60)

# CartPole Environment
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

# Generate data for different conditions
def generate_data(env, num_episodes=200, max_steps=30, condition='baseline'):
    """
    conditions:
    - baseline: random actions
    - random_intervention: random action perturbation
    - structured_intervention: always push right
    - action_freezing: always action=0
    """
    states = []
    actions = []
    next_states = []
    
    for _ in range(num_episodes):
        s = env.reset()
        
        for _ in range(max_steps):
            if condition == 'baseline':
                a = np.random.randint(0, 2)
            elif condition == 'random_intervention':
                # Random perturbation with some intervention
                a = np.random.randint(0, 2)
                if random.random() < 0.5:  # 50% intervention
                    a = 1 - a  # Flip action
            elif condition == 'structured_intervention':
                # Always push in one direction (structured intervention)
                a = 1
            elif condition == 'action_freezing':
                a = 0  # No control signal
            
            s_next = env.step(s, a)
            
            states.append(s.copy())
            actions.append(a)
            next_states.append(s_next.copy())
            
            s = s_next
    
    return np.array(states), np.array(actions), np.array(next_states)

env = CartPole()

# Generate data for each condition
conditions = ['baseline', 'random_intervention', 'structured_intervention', 'action_freezing']
data = {}

print('\nGenerating data for each condition...')
for cond in conditions:
    s, a, s_next = generate_data(env, condition=cond)
    data[cond] = {'states': s, 'actions': a, 'next_states': s_next}
    print(f'  {cond}: {len(s)} samples')

# Models
class BaselineModel(nn.Module):
    """Baseline: action concatenated"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 24), nn.ReLU(),
            nn.Linear(24, 8), nn.ReLU(),
            nn.Linear(8, 4))
    
    def forward(self, o, a):
        return self.net(torch.cat([o, a], dim=-1))

class CausalModel(nn.Module):
    """Causal: encoder-dynamics-decoder"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dynamics = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
    
    def forward(self, o, a):
        z = self.encoder(o)
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        return self.decoder(z_next), z

# Train and evaluate
def train_and_eval(states, actions, next_states):
    """Train causal model and evaluate variable emergence"""
    
    S_t = torch.FloatTensor(states)
    A_t = torch.FloatTensor(actions).float().unsqueeze(-1)
    S_next_t = torch.FloatTensor(next_states)
    
    # Train Causal model
    model = CausalModel()
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
    
    # Correlation with velocity (index 1) and angle (index 2)
    z_np = z[:, 1].numpy()  # Use first latent dim
    
    corr_vel = np.corrcoef(z_np, states[:, 1])[0, 1]
    corr_angle = np.corrcoef(z_np, states[:, 2])[0, 1]
    
    # Use absolute correlation
    corr = np.mean([abs(corr_vel), abs(corr_angle)])
    
    # Also compute MSE degradation (test shortcut vs true variable)
    # This is a more robust measure
    
    return corr, corr_vel, corr_angle

# Run experiments
print('\n' + '='*60)
print('RESULTS')
print('='*60)
print(f"{'Condition':<25} | {'|Corr|':>8} | {'Corr(v)':>8} | {'Corr(θ)':>8}")
print('-'*60)

results = {}
for cond in conditions:
    d = data[cond]
    corr, corr_vel, corr_angle = train_and_eval(d['states'], d['actions'], d['next_states'])
    results[cond] = {'corr': corr, 'corr_vel': corr_vel, 'corr_angle': corr_angle}
    print(f"{cond:<25} | {corr:>8.3f} | {corr_vel:>+8.3f} | {corr_angle:>+8.3f}")

print('='*60)

# Analysis
print('\n' + '='*60)
print('ANALYSIS')
print('='*60)

baseline = results['baseline']['corr']
random_int = results['random_intervention']['corr']
struct_int = results['structured_intervention']['corr']
freeze = results['action_freezing']['corr']

print(f'\nBaseline:                |Corr| = {baseline:.3f}')
print(f'Random Intervention:     |Corr| = {random_int:.3f} ({random_int-baseline:+.3f})')
print(f'Structured Intervention: |Corr| = {struct_int:.3f} ({struct_int-baseline:+.3f})')
print(f'Action Freezing:          |Corr| = {freeze:.3f} ({freeze-baseline:+.3f})')

print('\n' + '='*60)
print('CONCLUSION')
print('='*60)

if struct_int > baseline + 0.1:
    print('✓ Structured intervention significantly improves variable emergence!')
    print('  → Intervention drives variable formation')
elif random_int > baseline + 0.05:
    print('✓ Random intervention helps - data diversity matters')
else:
    print('→ Variables mainly from prediction + compression')
    
if freeze < baseline - 0.1:
    print('✓ Action freezing hurts - control signal important for variable discovery')
else:
    print('→ Control signal has limited effect')
