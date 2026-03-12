"""
Pendulum Analysis - Why does Causal architecture fail?
Analyzes: Task Controllability vs Variable Emergence

Run: python pendulum_analysis.py
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
print('PENDULUM ANALYSIS - CONTROLLABILITY')
print('='*60)

# Pendulum
class Pendulum:
    def __init__(self):
        self.g = 9.8
        self.l = 1.0
        self.damping = 0.1
        self.dt = 0.05
        
    def step(self, state, action):
        theta, omega = state
        torque = (action - 0.5) * 2
        omega_next = omega + (-self.g/self.l * np.sin(theta) - self.damping * omega + torque) * self.dt
        theta_next = theta + omega_next * self.dt
        theta_next = ((theta_next + np.pi) % (2*np.pi)) - np.pi
        return np.array([theta_next, omega_next])

# CartPole (for comparison)
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

def analyze_controllability(env_name, env, state_dim, n_samples=2000):
    """Analyze how much action affects state transitions"""
    states = []
    actions = []
    next_states = []
    
    for _ in range(n_samples):
        if env_name == 'pendulum':
            state = np.random.uniform(-0.5, 0.5, size=2)
        else:
            state = np.random.uniform(-0.05, 0.05, size=4)
        
        for _ in range(10):
            action = np.random.randint(0, 2)
            next_state = env.step(state, action)
            states.append(state.copy())
            actions.append(action)
            next_states.append(next_state.copy())
            state = next_state
    
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    
    # Compute action effect
    action_effect = []
    for i in range(len(states)):
        a0_next = env.step(states[i], 0)
        a1_next = env.step(states[i], 1)
        effect = np.linalg.norm(a1_next - a0_next)
        action_effect.append(effect)
    
    avg_effect = np.mean(action_effect)
    
    # Compare to natural dynamics
    natural_change = []
    for i in range(len(states)):
        change = np.linalg.norm(next_states[i] - states[i])
        natural_change.append(change)
    
    avg_natural = np.mean(natural_change)
    
    # Controllability ratio
    ctrl_ratio = avg_effect / (avg_natural + 1e-6)
    
    return {
        'env': env_name,
        'action_effect': avg_effect,
        'natural_change': avg_natural,
        'controllability': ctrl_ratio
    }

# Analyze both environments
print('\nControllability Analysis:')
print('-'*60)

pendulum = Pendulum()
cartpole = CartPole()

p_result = analyze_controllability('pendulum', pendulum, 2)
c_result = analyze_controllability('cartpole', cartpole, 4)

print(f"Pendulum:")
print(f"  Action effect:    {p_result['action_effect']:.4f}")
print(f"  Natural change:  {p_result['natural_change']:.4f}")
print(f"  Controllability: {p_result['controllability']:.4f}")

print(f"\nCartPole:")
print(f"  Action effect:    {c_result['action_effect']:.4f}")
print(f"  Natural change:  {c_result['natural_change']:.4f}")
print(f"  Controllability: {c_result['controllability']:.4f}")

# Train and compare
print('\n' + '='*60)
print('TRAINING COMPARISON')
print('='*60)

def train_and_test(env, state_dim, env_name):
    # Generate data
    states, actions, next_states = [], [], []
    for _ in range(200):
        if env_name == 'pendulum':
            state = np.random.uniform(-0.5, 0.5, size=2)
        else:
            state = np.random.uniform(-0.05, 0.05, size=4)
        
        for _ in range(30):
            action = np.random.randint(0, 2)
            next_state = env.step(state, action)
            states.append(state.copy())
            actions.append(action)
            next_states.append(next_state.copy())
            state = next_state
    
    S_t = torch.FloatTensor(np.array(states))
    A_t = torch.FloatTensor(np.array(actions)).float().unsqueeze(-1)
    S_next_t = torch.FloatTensor(np.array(next_states))
    
    # Models
    class Baseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + 1, 24), nn.ReLU(),
                nn.Linear(24, 16), nn.ReLU(),
                nn.Linear(16, state_dim))
        def forward(self, o, a):
            return self.net(torch.cat([o, a], dim=-1))
    
    class Causal(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(state_dim, 24), nn.ReLU(), nn.Linear(24, 4))
            self.dynamics = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
            self.decoder = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, state_dim))
        def forward(self, o, a):
            z = self.encoder(o)
            z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
            return self.decoder(z_next), z
    
    # Train Baseline
    m1 = Baseline()
    opt = torch.optim.Adam(m1.parameters(), lr=1e-3)
    for _ in range(15):
        idx = np.random.permutation(len(S_t))
        for i in range(0, len(idx), 32):
            loss = F.mse_loss(m1(S_t[idx[i:i+32]], A_t[idx[i:i+32]]), S_next_t[idx[i:i+32]])
            loss.backward(); opt.step(); opt.zero_grad()
    
    # Train Causal
    m2 = Causal()
    opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
    for _ in range(15):
        idx = np.random.permutation(len(S_t))
        for i in range(0, len(idx), 32):
            pred, z = m2(S_t[idx[i:i+32]], A_t[idx[i:i+32]])
            loss = F.mse_loss(pred, S_next_t[idx[i:i+32]])
            loss.backward(); opt.step(); opt.zero_grad()
    
    # Evaluate
    m1.eval()
    m2.eval()
    
    with torch.no_grad():
        z1 = m1.net[:3](torch.cat([S_t, A_t], dim=-1))
        _, z2 = m2(S_t, A_t)
    
    corr1 = np.mean([abs(np.corrcoef(z1[:, i].numpy(), np.array(states)[:, i])[0, 1]) 
                     for i in range(min(2, state_dim))])
    corr2 = np.mean([abs(np.corrcoef(z2[:, i].numpy(), np.array(states)[:, i])[0, 1]) 
                     for i in range(min(2, state_dim))])
    
    return corr1, corr2

print('\nTraining Pendulum...')
p_corr1, p_corr2 = train_and_test(pendulum, 2, 'pendulum')

print('Training CartPole...')
c_corr1, c_corr2 = train_and_test(cartpole, 4, 'cartpole')

print('\n' + '='*60)
print('RESULTS')
print('='*60)
print(f"{'Environment':<12} | {'Baseline':>10} | {'Causal':>10} | {'C-B':>10}")
print('-'*60)
print(f"{'Pendulum':<12} | {p_corr1:>10.3f} | {p_corr2:>10.3f} | {p_corr2-p_corr1:>+10.3f}")
print(f"{'CartPole':<12} | {c_corr1:>10.3f} | {c_corr2:>10.3f} | {c_corr2-c_corr1:>+10.3f}")

print('\n' + '='*60)
print('CONCLUSION')
print('='*60)
print(f'Pendulum controllability:   {p_result["controllability"]:.4f}')
print(f'CartPole controllability:   {c_result["controllability"]:.4f}')
print()
if p_result['controllability'] < c_result['controllability']:
    print('=> Lower controllability correlates with Causal failure')
    print('=> Variable emergence requires sufficient task controllability')
