#!/usr/bin/env python3
"""Fine-grained entropy scan: very low λ"""

import numpy as np

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


def train(lam, steps=3000):
    W = np.random.randn(64, 6) * 0.1
    h = np.zeros(64)
    entropies = []
    
    for _ in range(steps):
        ball = Ball()
        ball.reset()
        traj = [ball.pos.copy()]
        for _ in range(12):
            ball.step()
            traj.append(ball.pos.copy())
        traj = np.array(traj)
        
        x, y = traj[:3].flatten(), traj[-1]
        h = np.tanh(W @ x)
        e = y - (W.T @ h)[:2]
        W += 0.01 * (np.mean(e) * np.mean(h) - lam * np.sign(W))
        
        h_bin = (h > 0).astype(float)
        p1 = np.clip(np.mean(h_bin), 1e-10, 1-1e-10)
        ent = -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)
        entropies.append(ent)
    
    return np.mean(entropies[-500:])


print("Fine-grained entropy scan (very low λ):")
print("="*40)

lams = [0, 0.001, 0.002, 0.003, 0.004, 0.005]
ents = []

for lam in lams:
    ent = train(lam)
    ents.append(ent)
    print(f"λ = {lam:.3f}: Entropy = {ent:.4f}")

print("="*40)
print("\nEntropy drop analysis:")
for i in range(1, len(ents)):
    delta = ents[i] - ents[i-1]
    print(f"{lams[i-1]:.3f} → {lams[i]:.3f}: Δ = {delta:+.4f}")

# Check for sharp drop
print("\nSharp drop detection:")
if abs(ents[-1] - ents[0]) > 0.1:
    print("✓ Significant entropy drop detected!")
else:
    print("✗ No sharp drop")
