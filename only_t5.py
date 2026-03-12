"""
ULTIMATE TEST - ONLY t5 (overlap) - NO color info at all!
If model can predict from ONLY t5, it must have learned identity
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("ONLY t5 - NO COLOR INFO AT ALL")
print("="*60)

def generate_sequence():
    # Random colors (for identity marking only)
    color_a = [random.random(), random.random(), random.random()]
    color_b = [random.random(), random.random(), random.random()]
    
    x_a, y_a = random.uniform(5, 12), random.uniform(8, 24)
    x_b, y_b = random.uniform(20, 27), random.uniform(8, 24)
    
    vx_a, vy_a = random.uniform(-2, 2), random.uniform(-2, 2)
    vx_b, vy_b = random.uniform(-2, 2), random.uniform(-2, 2)
    
    # t0 colored (but we WON'T give this to model!)
    # Only give frames AFTER color is gone
    
    white = [1, 1, 1]
    
    # t1-t4: white balls
    for _ in range(1, 5):
        x_a, y_a = x_a + vx_a * 0.5, y_a + vy_a * 0.5
        x_b, y_b = x_b + vx_b * 0.5, y_b + vy_b * 0.5
        if x_a < 3 or x_a > 29: vx_a *= -1
        if y_a < 3 or y_a > 29: vy_a *= -1
        x_b = ((x_b - 3) % 26) + 3
        y_b = ((y_b - 3) % 26) + 3
    
    # t5: overlap (we give THIS to model)
    img_t5 = np.zeros((32, 32, 3), dtype=np.float32)
    img_t5[16, 16] = white
    
    # t6-t9: more white
    for step in range(6, 10):
        if step == 7:
            if random.random() < 0.5: vx_a *= -1
            if random.random() < 0.5: vy_a *= -1
            vx_b *= random.uniform(0.5, 1.5)
            vy_b *= random.uniform(0.5, 1.5)
        
        x_a, y_a = x_a + vx_a * 0.5, y_a + vy_a * 0.5
        x_b, y_b = x_b + vx_b * 0.5, y_b + vy_b * 0.5
        if x_a < 3 or x_a > 29: vx_a *= -1
        if y_a < 3 or y_a > 29: vy_a *= -1
        x_b = ((x_b - 3) % 26) + 3
        y_b = ((y_b - 3) % 26) + 3
    
    # Target: ball A position (unknown which one!)
    target = [x_a / 32, y_a / 32]
    
    return img_t5, target

def generate_data(n=5000):
    frames, targets = [], []
    for _ in range(n):
        f, t = generate_sequence()
        frames.append(f)
        targets.append(t)
    return np.array(frames), np.array(targets)

print("\n1. Generating data...")
X, Y = generate_data(5000)
X = torch.FloatTensor(X).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)
print(f"   X: {X.shape}, Y: {Y.shape}")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

print("\n2. Training on t5 ONLY...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(20):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p = m(X[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    mse = F.mse_loss(m(X), Y).item()

print(f"\n   MSE (t5 ONLY): {mse:.4f}")

# Compare: what's the BEST possible MSE?
# Random guessing would give:
random_mse = ((Y - Y.mean())**2).mean().item()
print(f"   Random guess MSE: {random_mse:.4f}")
print(f"   If MSE ~ random_guess: model learned NOTHING")
print(f"   If MSE << random_guess: model found a pattern!")
