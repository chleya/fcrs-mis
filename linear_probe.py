#!/usr/bin/env python3
"""Linear Probe Test: Can we decode velocity from latent?"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

np.random.seed(42)

class Ball:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * 10 + 3
        self.vel = (np.random.rand(2) - 0.5) * 0.4
    
    def step(self):
        self.vel += (np.random.rand(2) - 0.5) * 0.1
        self.vel = np.clip(self.vel, -0.8, 0.8)
        self.pos += self.vel
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)


def train_model(dim=32, T=20, lam=0.01, steps=3000):
    """Train model and collect latent-velocity pairs"""
    W = np.random.randn(dim, 6) * 0.1
    
    for _ in range(steps):
        ball = Ball()
        ball.reset()
        traj = [ball.pos.copy()]
        for _ in range(T + 5):
            ball.step()
            traj.append(ball.pos.copy())
        traj = np.array(traj)
        
        x = traj[:3].flatten()
        y = traj[3 + T]
        
        h = np.tanh(W @ x)
        e = y - (W.T @ h)[:2]
        W += 0.01 * (np.mean(e) * np.mean(h) - lam * np.sign(W))
        W = np.clip(W, -10, 10)
    
    # Collect data
    h_list, v_list, p_list = [], [], []
    
    for _ in range(1000):
        ball = Ball()
        ball.reset()
        traj = [ball.pos.copy()]
        for _ in range(T + 5):
            ball.step()
            traj.append(ball.pos.copy())
        traj = np.array(traj)
        
        x = traj[:3].flatten()
        h = np.tanh(W @ x)
        
        v = (traj[3 + T] - traj[2 + T - 1])
        p = traj[3 + T]
        
        h_list.append(h)
        v_list.append(v)
        p_list.append(p)
    
    return np.array(h_list), np.array(v_list), np.array(p_list)


def linear_probe(X, y, name="target"):
    """Train linear probe and evaluate"""
    model = LinearRegression()
    
    # Cross-validation
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        r2 = np.mean(scores)
        r2_std = np.std(scores)
    except:
        r2, r2_std = 0, 0
    
    # Full fit for coefficients
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Correlation
    corr = np.corrcoef(y.flatten(), y_pred.flatten())[0, 1]
    
    return {
        "name": name,
        "r2": r2,
        "r2_std": r2_std,
        "corr": corr
    }


# Main test
print("="*60)
print("LINEAR PROBE TEST")
print("="*60)

H, V, P = train_model(dim=32, T=20, lam=0.01)

print(f"\nData shape: H={H.shape}, V={V.shape}, P={P.shape}")

# Probe velocity
print("\n1. Probing velocity...")
v_probe = linear_probe(H, V, "velocity")
print(f"   R^2 = {v_probe['r2']:.3f} +/- {v_probe['r2_std']:.3f}")
print(f"   Correlation = {v_probe['corr']:.3f}")

# Probe velocity components
print("\n2. Probing velocity_x...")
vx_probe = linear_probe(H, V[:, 0:1], "velocity_x")
print(f"   R² = {vx_probe['r2']:.3f} ± {vx_probe['r2_std']:.3f}")

print("\n3. Probing velocity_y...")
vy_probe = linear_probe(H, V[:, 1:2], "velocity_y")
print(f"   R² = {vy_probe['r2']:.3f} ± {vy_probe['r2_std']:.3f}")

# Probe position
print("\n4. Probing position...")
p_probe = linear_probe(H, P, "position")
print(f"   R² = {p_probe['r2']:.3f} ± {p_probe['r2_std']:.3f}")
print(f"   Correlation = {p_probe['corr']:.3f}")

# Probe velocity magnitude
print("\n5. Probing velocity magnitude...")
v_mag = np.sqrt(V[:, 0]**2 + V[:, 1]**2).reshape(-1, 1)
vm_probe = linear_probe(H, v_mag, "velocity_magnitude")
print(f"   R² = {vm_probe['r2']:.3f} ± {vm_probe['r2_std']:.3f}")

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"{'Target':<20} | {'R²':>8} | {'Corr':>8}")
print("-"*60)
print(f"{'velocity (2D)':<20} | {v_probe['r2']:>8.3f} | {v_probe['corr']:>8.3f}")
print(f"{'position (2D)':<20} | {p_probe['r2']:>8.3f} | {p_probe['corr']:>8.3f}")
print(f"{'velocity magnitude':<20} | {vm_probe['r2']:>8.3f} | {'--':>8}")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)

if v_probe['r2'] > 0.6:
    print(">>> velocity STRONGLY encoded (R² > 0.6)")
elif v_probe['r2'] > 0.3:
    print(">>> velocity PARTIALLY encoded (0.3 < R² < 0.6)")
elif v_probe['r2'] > 0.1:
    print(">>> velocity WEAKLY encoded (0.1 < R² < 0.3)")
else:
    print(">>> velocity NOT meaningfully encoded (R² < 0.1)")

if p_probe['r2'] > v_probe['r2']:
    print(f">>> position dominates (R²={p_probe['r2']:.3f} > {v_probe['r2']:.3f})")
else:
    print(f">>> velocity dominates (R²={v_probe['r2']:.3f} > {p_probe['r2']:.3f})")
