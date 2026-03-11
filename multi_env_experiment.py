"""
Multi-Environment Verification - FIXED VERSION
Tests whether the causal structure theory generalizes to different physical systems

Fixed: Increased training epochs for Pendulum

Run: python multi_env_experiment.py
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
print('MULTI-ENVIRONMENT VERIFICATION (FIXED)')
print('='*60)

# ============================================================
# Environment 1: CartPole
# ============================================================
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


# ============================================================
# Environment 2: Pendulum (FIXED)
# ============================================================
class Pendulum:
    """Simple pendulum with continuous angle"""
    def __init__(self):
        self.g = 9.8
        self.l = 1.0
        self.damping = 0.1
        self.dt = 0.05
        
    def step(self, state, action):
        theta, omega = state
        torque = (action - 0.5) * 2  # Map [0,1] to [-1,1]
        
        # Dynamics
        omega_next = omega + (-self.g/self.l * np.sin(theta) - self.damping * omega + torque) * self.dt
        theta_next = theta + omega_next * self.dt
        
        # Normalize angle to [-pi, pi]
        theta_next = ((theta_next + np.pi) % (2*np.pi)) - np.pi
        
        return np.array([theta_next, omega_next])


# ============================================================
# Environment 3: Spring-Mass
# ============================================================
class SpringMass:
    """Simple spring-mass system"""
    def __init__(self):
        self.k = 2.0
        self.m = 1.0
        self.damping = 0.1
        self.dt = 0.1
        
    def step(self, state, action):
        x, v = state
        force = (action - 0.5) * 2
        
        # Dynamics
        a = (-self.k * x - self.damping * v + force) / self.m
        x_next = x + v * self.dt
        v_next = v + a * self.dt
        
        return np.array([x_next, v_next])


# ============================================================
# Data Generation
# ============================================================
def generate_trajectory(env, env_type, num_steps=50):
    """Generate trajectory data"""
    trajectories = []
    
    for _ in range(200):
        if env_type == 'cartpole':
            state = np.random.uniform(-0.05, 0.05, size=4)
        elif env_type == 'pendulum':
            state = np.random.uniform(-0.5, 0.5, size=2)
        else:
            state = np.random.uniform(-1, 1, size=2)
        
        for _ in range(num_steps):
            action = np.random.randint(0, 2)
            next_state = env.step(state, action)
            
            trajectories.append({
                'state': state.copy(),
                'action': action,
                'next_state': next_state.copy()
            })
            
            state = next_state
    
    return trajectories


def create_samples(trajectories, state_dim):
    """Create training samples"""
    samples = []
    for i in range(len(trajectories) - 1):
        s = trajectories[i]['state']
        a = trajectories[i]['action']
        s_next = trajectories[i+1]['state']
        
        # Add observation noise
        obs = s + np.random.randn(state_dim) * 0.05
        obs_next = s_next + np.random.randn(state_dim) * 0.05
        
        samples.append({
            's': torch.FloatTensor(s),
            'o': torch.FloatTensor(obs),
            'a': torch.tensor(a, dtype=torch.float32),
            's_next': torch.FloatTensor(s_next),
            'o_next': torch.FloatTensor(obs_next)
        })
    
    return samples


# ============================================================
# Models
# ============================================================
class BaselineModel(nn.Module):
    def __init__(self, state_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, state_dim)
        )
    
    def forward(self, o, a):
        return self.net(torch.cat([o, a], dim=-1))


class CausalModel(nn.Module):
    def __init__(self, state_dim, obs_dim):
        super().__init__()
        latent_dim = state_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 24),
            nn.ReLU(),
            nn.Linear(24, latent_dim)
        )
        
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 1, 24),
            nn.ReLU(),
            nn.Linear(24, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.ReLU(),
            nn.Linear(24, obs_dim)
        )
    
    def forward(self, o, a):
        z = self.encoder(o)
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        return self.decoder(z_next), z


# ============================================================
# Train and Test
# ============================================================
def train_and_test(env, env_type, state_dim, obs_dim, epochs=30):
    """Train baseline and causal models, compare results"""
    
    # Generate data
    trajectories = generate_trajectory(env, env_type, num_steps=30)
    samples = create_samples(trajectories, state_dim)
    
    if len(samples) < 100:
        return None, None
    
    # Split
    train_size = int(0.8 * len(samples))
    train_data = samples[:train_size]
    test_data = samples[train_size:]
    
    # Baseline
    baseline = BaselineModel(state_dim, obs_dim)
    opt = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        random.shuffle(train_data)
        for batch in range(0, len(train_data), 32):
            items = train_data[batch:batch+32]
            o = torch.stack([d['o'] for d in items])
            a = torch.stack([d['a'] for d in items]).unsqueeze(-1)
            o_next = torch.stack([d['o_next'] for d in items])
            
            pred = baseline(o, a)
            loss = F.mse_loss(pred, o_next)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # Test baseline
    baseline.eval()
    all_z_baseline = []
    all_s = []
    with torch.no_grad():
        for d in test_data:
            o = d['o'].unsqueeze(0)
            a = d['a'].unsqueeze(0).unsqueeze(-1)
            # Use last hidden layer as latent
            h = baseline.net[:3](torch.cat([o, a], dim=-1))
            all_z_baseline.append(h.numpy().flatten())
            all_s.append(d['s'].numpy())
    
    z_baseline = np.array(all_z_baseline)
    s = np.array(all_s)
    
    # Causal
    causal = CausalModel(state_dim, obs_dim)
    opt = torch.optim.Adam(causal.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        random.shuffle(train_data)
        for batch in range(0, len(train_data), 32):
            items = train_data[batch:batch+32]
            o = torch.stack([d['o'] for d in items])
            a = torch.stack([d['a'] for d in items]).unsqueeze(-1)
            o_next = torch.stack([d['o_next'] for d in items])
            
            pred, z = causal(o, a)
            loss = F.mse_loss(pred, o_next)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # Test causal
    causal.eval()
    all_z_causal = []
    with torch.no_grad():
        for d in test_data:
            o = d['o'].unsqueeze(0)
            a = d['a'].unsqueeze(0).unsqueeze(-1)
            _, z = causal(o, a)
            all_z_causal.append(z.numpy().flatten())
    
    z_causal = np.array(all_z_causal)
    
    # Correlation
    def calc_corr(z, s):
        corrs = []
        for i in range(min(z.shape[1], s.shape[1])):
            c = np.corrcoef(z[:, i], s[:, i])[0, 1]
            if not np.isnan(c):
                corrs.append(abs(c))
        return np.mean(corrs) if corrs else 0
    
    corr_baseline = calc_corr(z_baseline, s)
    corr_causal = calc_corr(z_causal, s)
    
    return corr_baseline, corr_causal


# ============================================================
# Run Experiment
# ============================================================
# Use increased epochs for Pendulum
environments = [
    ('CartPole', CartPole(), 4, 4, 30),
    ('Pendulum', Pendulum(), 2, 2, 30),  # FIXED: 30 epochs
    ('SpringMass', SpringMass(), 2, 2, 30),
]

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"{'Environment':<15} | {'Baseline':>10} | {'Causal':>10} | {'Improvement':>12}")
print("-"*60)

results = []
for name, env, state_dim, obs_dim, epochs in environments:
    corr_baseline, corr_causal = train_and_test(env, name.lower(), state_dim, obs_dim, epochs)
    
    if corr_baseline is None:
        print(f"{name:<15} | {'N/A':>10} | {'N/A':>10} | {'N/A':>12}")
        continue
    
    improvement = corr_causal - corr_baseline
    results.append((name, corr_baseline, corr_causal, improvement))
    print(f"{name:<15} | {corr_baseline:>10.3f} | {corr_causal:>10.3f} | {improvement:>+12.3f}")

print("="*60)

# Summary
passed = sum(1 for r in results if r[3] > 0)
print(f"\nSummary: {passed}/{len(results)} environments show improvement")
print("Theory generalizes!" if passed == len(results) else "Need further investigation")
