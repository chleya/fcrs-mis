"""
COMBINATORIAL SCALING EXPERIMENT
Train on 2 balls, test on 6 balls

Key: Trajectory models typically fail when object count changes dramatically
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
print("COMBINATORIAL SCALING EXPERIMENT")
print("="*60)

def clamp(val):
    return max(3, min(28, int(val)))

def generate_train(n=2000, n_objects=2):
    """Generate training data with N objects"""
    X, Y = [], []
    for _ in range(n):
        positions = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        # t0
        img0 = np.zeros((32, 32, 3), np.float32)
        for i, (x, y) in enumerate(positions):
            if i == 0:
                img0[clamp(y), clamp(x)] = [1, 0, 0]
            else:
                img0[clamp(y), clamp(x)] = [0, 0, 1]
        
        # t10
        for step in range(10):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx * 0.5, y + vy * 0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i] = (x, y)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img10[clamp(y), clamp(x)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(positions[0][0] / 32)
    
    return np.array(X), np.array(Y)

def generate_test(n=1000, n_objects=6):
    """Generate test data with MORE objects"""
    X, Y = [], []
    for _ in range(n):
        positions = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        # t0
        img0 = np.zeros((32, 32, 3), np.float32)
        for i, (x, y) in enumerate(positions):
            if i == 0:
                img0[clamp(y), clamp(x)] = [1, 0, 0]
            else:
                img0[clamp(y), clamp(x)] = [0, 0, 1]
        
        # t10
        for step in range(10):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx * 0.5, y + vy * 0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i] = (x, y)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img10[clamp(y), clamp(x)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(positions[0][0] / 32)
    
    return np.array(X), np.array(Y)

print("\n1. Generating data...")
X_train, Y_train = generate_train(2000, 2)
X_test_2, Y_test_2 = generate_test(1000, 2)
X_test_6, Y_test_6 = generate_test(1000, 6)

X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
X_test_2 = torch.FloatTensor(X_test_2).permute(0, 3, 1, 2)
X_test_6 = torch.FloatTensor(X_test_6).permute(0, 3, 1, 2)
Y_train = torch.FloatTensor(Y_train)
Y_test_2 = torch.FloatTensor(Y_test_2)
Y_test_6 = torch.FloatTensor(Y_test_6)

print(f"   Train (2 obj): {X_train.shape}")
print(f"   Test (2 obj): {X_test_2.shape}")
print(f"   Test (6 obj): {X_test_6.shape}")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(self.enc(x).flatten(1)).squeeze()

print("\n2. Training on 2 objects...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        p = m(X_train[idx[i:i+64]])
        loss = F.mse_loss(p, Y_train[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

# Test
m.eval()
with torch.no_grad():
    mse_test_2 = F.mse_loss(m(X_test_2), Y_test_2).item()
    mse_test_6 = F.mse_loss(m(X_test_6), Y_test_6).item()

random_mse_2 = Y_test_2.var().item()
random_mse_6 = Y_test_6.var().item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Train on 2 objects:")
print(f"  Test (2 objects): MSE = {mse_test_2:.4f} ({(random_mse_2-mse_test_2)/random_mse_2*100:.1f}% < random)")
print(f"  Test (6 objects): MSE = {mse_test_6:.4f} ({(random_mse_6-mse_test_6)/random_mse_6*100:.1f}% < random)")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
drop = (mse_test_6 - mse_test_2) / mse_test_2 * 100
print(f"Performance drop from 2→6 objects: {drop:.1f}%")

if drop > 50:
    print("=> Trajectory model FAILS on novel object count!")
    print("=> This is where object representation could help")
else:
    print("=> Model generalizes reasonably to more objects")
