"""
Ablation Experiment
Tests the necessity of each component in the causal architecture

Components to ablate:
1. Full Causal: encoder + dynamics + decoder (baseline)
2. No Dynamics: encoder + direct prediction
3. No Encoder Isolation: encoder receives action as input
4. No Decoder: predict directly from latent

Run: python ablation_experiment.py
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
print('ABLATION EXPERIMENT')
print('='*60)

# ============================================================
# Data Generation (same as mini_experiment)
# ============================================================
class SimpleCartPole:
    def __init__(self):
        self.gravity = 9.8
        self.length = 0.5
        self.force_mag = 10.0
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

def generate_data(num=2000, max_steps=30):
    env = SimpleCartPole()
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

print("Generating data...")
data = generate_data(2000)
print(f"Samples: {len(data)}")

# Split
train_data = data[:1600]
test_data = data[1600:]

# ============================================================
# Model Variants
# ============================================================

class FullCausal(nn.Module):
    """Full causal: encoder -> dynamics -> decoder"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dynamics = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))  # z + a
        self.decoder = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 16))
    
    def forward(self, o, a):
        z = self.encoder(o)
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        return self.decoder(z_next), z


class NoDynamics(nn.Module):
    """No dynamics: encoder -> direct prediction (skip dynamics)"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 4))
        self.predictor = nn.Sequential(nn.Linear(4+1, 24), nn.ReLU(), nn.Linear(24, 16))  # z + a directly
    
    def forward(self, o, a):
        z = self.encoder(o)
        return self.predictor(torch.cat([z, a], dim=-1)), z


class NoEncoderIsolation(nn.Module):
    """No encoder isolation: encoder receives action"""
    def __init__(self):
        super().__init__()
        # Encoder receives both observation AND action
        self.encoder = nn.Sequential(nn.Linear(17, 24), nn.ReLU(), nn.Linear(24, 4))  # o + a
        self.dynamics = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 16))
    
    def forward(self, o, a):
        z = self.encoder(torch.cat([o, a], dim=-1))  # action leaks into encoder
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        return self.decoder(z_next), z


class NoDecoder(nn.Module):
    """No decoder: predict directly from latent"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dynamics = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        # Directly output prediction from z_next (no decoder)
        self.predictor = nn.Linear(4, 16)
    
    def forward(self, o, a):
        z = self.encoder(o)
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        return self.predictor(z_next), z


# ============================================================
# Training Function
# ============================================================

def train_model(model, train_data, epochs=8, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        random.shuffle(train_data)
        for batch in range(0, len(train_data), 32):
            batch_data = train_data[batch:batch+32]
            o = torch.stack([d["o_t"] for d in batch_data])
            a = torch.stack([d["a_t"] for d in batch_data]).unsqueeze(-1)
            o_next = torch.stack([d["o_t1"] for d in batch_data])
            
            if isinstance(model, FullCausal) or isinstance(model, NoDecoder):
                pred, z = model(o, a)
            else:
                pred, z = model(o, a)
            
            loss = F.mse_loss(pred, o_next)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


def evaluate_model(model, test_data):
    model.eval()
    all_z, all_v = [], []
    
    with torch.no_grad():
        for d in test_data:
            o = d["o_t"].unsqueeze(0)
            a = d["a_t"].unsqueeze(0).unsqueeze(-1)
            
            if isinstance(model, FullCausal) or isinstance(model, NoDecoder):
                pred, z = model(o, a)
            else:
                pred, z = model(o, a)
            
            all_z.append(z.numpy().flatten())
            all_v.append(d["v_t"].numpy())
    
    z = np.array(all_z)
    v = np.array(all_v)
    
    # Correlation for each dimension
    corr_angle = np.corrcoef(z[:, 2], v[:, 2])[0, 1]
    corr_vel = np.corrcoef(z[:, 1], v[:, 1])[0, 1]
    
    return corr_angle, corr_vel


# ============================================================
# Run Ablation
# ============================================================

models = [
    ("Full Causal", FullCausal()),
    ("No Dynamics", NoDynamics()),
    ("No Encoder Isolation", NoEncoderIsolation()),
    ("No Decoder", NoDecoder()),
]

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"{'Model':<25} | {'Corr(angle)':>12} | {'Corr(velocity)':>12}")
print("-"*60)

results = []
for name, model in models:
    model = train_model(model, train_data)
    corr_a, corr_v = evaluate_model(model, test_data)
    results.append((name, corr_a, corr_v))
    print(f"{name:<25} | {corr_a:>+12.3f} | {corr_v:>+12.3f}")

print("="*60)

# ============================================================
# Analysis
# ============================================================
print("\nANALYSIS:")
print("-"*60)

baseline_corr = results[0][1]  # Full causal
print(f"Baseline (Full Causal): Corr(angle) = {baseline_corr:+.3f}")

for i, (name, corr_a, corr_v) in enumerate(results[1:], 1):
    change = corr_a - baseline_corr
    print(f"{name}: {change:+.3f} (vs baseline)")
    if abs(change) < 0.1:
        print(f"  → Component NOT critical")
    else:
        print(f"  → Component IS critical")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
