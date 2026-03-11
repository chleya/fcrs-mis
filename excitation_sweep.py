"""
Excitation Intensity Sweep Experiment
Tests the relationship between action force magnitude and variable emergence

This experiment tests whether there's an optimal excitation level
(inspired by persistent excitation theory in system identification)

Expected: Inverted-U relationship
- Too small: signal < noise
- Optimal: maximum information
- Too large: dynamics too fast

Run: python excitation_sweep.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print('='*60)
print('EXCITATION INTENSITY SWEEP')
print('='*60)

# ============================================================
# CartPole with variable force
# ============================================================
class SimpleCartPole:
    def __init__(self, force_mag=10.0):
        self.force_mag = force_mag
        self.gravity = 9.8
        self.length = 0.5
        self.tau = 0.02
        self.x_threshold = 2.4
        self.theta_threshold = 0.20944
    
    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=(4,))
        return self.state.copy()
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = force / 1.1 + 0.05 * theta_dot**2 * sintheta
        thetaacc = (9.8*sintheta - costheta*temp) / (self.length * (4/3 - 0.1*costheta**2/1.1))
        xacc = temp - 0.05*thetaacc*costheta/1.1
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        done = abs(x) > 2.4 or abs(theta) > 0.20944
        return self.state.copy(), done


def generate_data(force_mag, num=1000, max_steps=20):
    """Generate data with specified force magnitude"""
    env = SimpleCartPole(force_mag=force_mag)
    obs_transform = np.random.randn(4, 16) * 0.1
    
    data = []
    for _ in range(num):
        v = env.reset()
        for _ in range(max_steps):
            a = np.random.randint(0, 2)
            o = v @ obs_transform + np.random.randn(16) * 0.05
            v_next, done = env.step(a)
            data.append((v.copy(), o.copy(), a))
            v = v_next
            if done: break
    
    samples = []
    for i in range(len(data) - 1):
        samples.append({
            "v_t": torch.FloatTensor(data[i][0]),
            "o_t": torch.FloatTensor(data[i][1]),
            "a_t": torch.tensor(data[i][2], dtype=torch.float32),
            "v_t1": torch.FloatTensor(data[i+1][0]),
            "o_t1": torch.FloatTensor(data[i+1][1]),
        })
    return samples


class CausalModel(nn.Module):
    """Causal architecture"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dynamics = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 16))
    
    def forward(self, o, a):
        z = self.encoder(o)
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        return self.decoder(z_next), z


def train_and_test(data, epochs=8):
    """Train causal model and test correlation"""
    model = CausalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for _ in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), 32):
            batch = data[i:i+32]
            o = torch.stack([d["o_t"] for d in batch])
            a = torch.stack([d["a_t"] for d in batch]).unsqueeze(-1)
            o_next = torch.stack([d["o_t1"] for d in batch])
            
            pred, z = model(o, a)
            loss = F.mse_loss(pred, o_next)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Test
    model.eval()
    all_z, all_v = [], []
    with torch.no_grad():
        for d in data[:500]:  # Use subset for testing
            o = d["o_t"].unsqueeze(0)
            a = d["a_t"].unsqueeze(0).unsqueeze(-1)
            _, z = model(o, a)
            all_z.append(z.numpy().flatten())
            all_v.append(d["v_t"].numpy())
    
    z = np.array(all_z)
    v = np.array(all_v)
    
    corr_angle = np.corrcoef(z[:, 2], v[:, 2])[0, 1]
    corr_vel = np.corrcoef(z[:, 1], v[:, 1])[0, 1]
    
    return corr_angle, corr_vel


# ============================================================
# Run Excitation Sweep
# ============================================================
forces = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

print("\nForce | Samples | Corr(angle) | Corr(velocity) | |Corr|")
print("-"*65)

results = []
for force in forces:
    # Generate data
    data = generate_data(force, num=1000)
    samples = len(data)
    
    if samples < 100:
        print(f"{force:5.1f} | {samples:7} | (too few samples)")
        continue
    
    # Train and test
    corr_a, corr_v = train_and_test(data)
    abs_corr = np.mean([abs(corr_a), abs(corr_v)])
    
    results.append((force, corr_a, corr_v, abs_corr))
    print(f"{force:5.1f} | {samples:7} | {corr_a:+10.3f} | {corr_v:+13.3f} | {abs_corr:.3f}")

print("="*60)

# ============================================================
# Analysis
# ============================================================
print("\nANALYSIS:")
print("-"*60)

# Find optimal
best = max(results, key=lambda x: x[3])
print(f"Best excitation: Force = {best[0]}")
print(f"  Corr(angle) = {best[1]:+.3f}")
print(f"  Corr(velocity) = {best[2]:+.3f}")
print(f"  |Corr| = {best[3]:.3f}")

# Check for inverted-U relationship
print("\n|Corr| vs Force:")
for force, ca, cv, ac in results:
    bar = "█" * int(ac * 20)
    print(f"  {force:5.1f}: {ac:.3f} {bar}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("Testing whether excitation level affects variable emergence...")
