"""
ULTIMATE Identity Test
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
print("ULTIMATE IDENTITY TEST")
print("="*60)

def generate_sequence():
    color_a = [random.random(), random.random(), random.random()]
    color_b = [random.random(), random.random(), random.random()]
    
    x_a, y_a = random.uniform(5, 12), random.uniform(8, 24)
    x_b, y_b = random.uniform(20, 27), random.uniform(8, 24)
    
    vx_a, vy_a = random.uniform(-2, 2), random.uniform(-2, 2)
    vx_b, vy_b = random.uniform(-2, 2), random.uniform(-2, 2)
    
    frames = []
    
    # t0: colored
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[int(y_a), int(x_a)] = color_a
    img[int(y_b), int(x_b)] = color_b
    frames.append(img.copy())
    
    white = [1, 1, 1]
    
    # t1-t4: white
    for _ in range(1, 5):
        x_a, y_a = x_a + vx_a * 0.5, y_a + vy_a * 0.5
        x_b, y_b = x_b + vx_b * 0.5, y_b + vy_b * 0.5
        if x_a < 3 or x_a > 29: vx_a *= -1
        if y_a < 3 or y_a > 29: vy_a *= -1
        x_b = ((x_b - 3) % 26) + 3
        y_b = ((y_b - 3) % 26) + 3
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[int(y_a), int(x_a)] = white
        img[int(y_b), int(x_b)] = white
        frames.append(img.copy())
    
    # t5: overlap
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[16, 16] = white
    frames.append(img.copy())
    
    # t6-t9
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
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[int(y_a), int(x_a)] = white
        img[int(y_b), int(x_b)] = white
        frames.append(img.copy())
    
    target = [x_a / 32, y_a / 32]
    return np.array(frames), target

def generate_data(n=5000):
    frames, targets = [], []
    for _ in range(n):
        f, t = generate_sequence()
        frames.append(f)
        targets.append(t)
    return np.array(frames), np.array(targets)

print("\n1. Generating data...")
X, Y = generate_data(5000)
X = torch.FloatTensor(X).permute(0, 1, 4, 2, 3)
Y = torch.FloatTensor(Y)

X_t0 = X[:, 0]
X_t5 = X[:, 5]

print(f"   X: {X.shape}, Y: {Y.shape}")

class Baseline(nn.Module):
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

print("\n2. Training...")

results = {}

# t0 only (has color)
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X_t0))
    for i in range(0, len(X_t0), 32):
        p = m(X_t0[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
results['t0'] = F.mse_loss(m(X_t0), Y).item()

# t5 only (overlap)
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X_t5))
    for i in range(0, len(X_t5), 32):
        p = m(X_t5[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
results['t5'] = F.mse_loss(m(X_t5), Y).item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
for name, mse in results.items():
    print(f"{name}: MSE = {mse:.4f}")
