"""
Spatial Position Cognition Test
1. Position regression: image -> (x, y)
2. Relative position: left/right classification
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
print("SPATIAL POSITION TEST")
print("="*60)

# 1. Position Regression
def generate_position_data(n=3000):
    images = []
    positions = []
    
    for _ in range(n):
        # Random ball position
        x = random.uniform(5, 27)
        y = random.uniform(5, 27)
        
        # Image
        img = np.zeros((32, 32, 3), dtype=np.float32)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx*dx + dy*dy <= 4:
                    px, py = int(x+dx), int(y+dy)
                    if 0 <= px < 32 and 0 <= py < 32:
                        img[py, px] = [1, 0, 0]
        
        images.append(img)
        positions.append([x/32, y/32])
    
    return np.array(images), np.array(positions)

# 2. Relative Position (binary)
def generate_relative_data(n=3000):
    images = []
    labels = []
    
    for _ in range(n):
        x1 = random.uniform(5, 27)
        x2 = random.uniform(5, 27)
        y1 = random.uniform(5, 27)
        y2 = random.uniform(5, 27)
        
        # Image with two balls
        img = np.zeros((32, 32, 3), dtype=np.float32)
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx*dx + dy*dy <= 4:
                    # Ball 1 (white)
                    px1, py1 = int(x1+dx), int(y1+dy)
                    if 0 <= px1 < 32 and 0 <= py1 < 32:
                        img[py1, px1] = [1, 1, 1]
                    
                    # Ball 2 (gray)
                    px2, py2 = int(x2+dx), int(y2+dy)
                    if 0 <= px2 < 32 and 0 <= py2 < 32:
                        img[py2, px2] = [0.5, 0.5, 0.5]
        
        # Label: 1 if ball1 is left of ball2
        label = 1 if x1 < x2 else 0
        images.append(img)
        labels.append(label)
    
    return np.array(images), np.array(labels)

print("\n1. Generating position regression data...")
X_pos, Y_pos = generate_position_data(3000)
X_pos = torch.FloatTensor(X_pos).permute(0, 3, 1, 2)
Y_pos = torch.FloatTensor(Y_pos)

print("2. Generating relative position data...")
X_rel, Y_rel = generate_relative_data(3000)
X_rel = torch.FloatTensor(X_rel).permute(0, 3, 1, 2)
Y_rel = torch.FloatTensor(Y_rel).float()

print(f"   Position: {X_pos.shape} -> {Y_pos.shape}")
print(f"   Relative: {X_rel.shape} -> {Y_rel.shape}")

# Model
class CNN(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*4*4, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Test 1: Position Regression
print("\n3. Testing position regression...")
m_pos = CNN(2)
opt = torch.optim.Adam(m_pos.parameters(), lr=1e-3)

for ep in range(10):
    idx = torch.randperm(len(X_pos))
    for i in range(0, len(X_pos), 32):
        p = m_pos(X_pos[idx[i:i+32]])
        loss = F.mse_loss(p, Y_pos[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()

m_pos.eval()
with torch.no_grad():
    pred_pos = m_pos(X_pos)
    mse_pos = F.mse_loss(pred_pos, Y_pos).item()

print(f"   Position MSE: {mse_pos:.6f}")

# Test 2: Relative Position
print("\n4. Testing relative position...")
m_rel = CNN(1)
opt = torch.optim.Adam(m_rel.parameters(), lr=1e-3)

for ep in range(10):
    idx = torch.randperm(len(X_rel))
    for i in range(0, len(X_rel), 32):
        p = m_rel(X_rel[idx[i:i+32]]).squeeze()
        loss = F.binary_cross_entropy_with_logits(p, Y_rel[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()

m_rel.eval()
with torch.no_grad():
    pred_rel = (torch.sigmoid(m_rel(X_rel)) > 0.5).float()
    acc_rel = (pred_rel.squeeze() == Y_rel).float().mean().item()

print(f"   Relative accuracy: {acc_rel:.2%}")

# Summary
print("\n" + "="*60)
print("SPATIAL COGNITION SUMMARY")
print("="*60)
print(f"Position regression MSE: {mse_pos:.6f}")
print(f"Relative position accuracy: {acc_rel:.2%}")

if mse_pos < 0.01:
    print("\n✓ Model CAN learn absolute position")
else:
    print("\n✗ Model struggles with position")
    
if acc_rel > 0.9:
    print("✓ Model CAN learn relative position")
else:
    print("✗ Model struggles with relative position")
