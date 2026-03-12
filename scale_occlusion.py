"""
Scale and Occlusion Test
1. Scale invariance: recognize object at different sizes
2. Occlusion: recognize partially hidden object
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
print("SCALE AND OCCLUSION TEST")
print("="*60)

def generate_scale_data(n=3000):
    """Test scale invariance"""
    images = []
    labels = []  # Size category: 0=small, 1=medium, 2=large
    
    for _ in range(n):
        size = random.choice([2, 4, 6])  # radius
        x = random.uniform(5, 27)
        y = random.uniform(5, 27)
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        
        # Draw ball with different sizes
        for dx in range(-size, size+1):
            for dy in range(-size, size+1):
                if dx*dx + dy*dy <= size*size:
                    px, py = int(x+dx), int(y+dy)
                    if 0 <= px < 32 and 0 <= py < 32:
                        img[py, px] = [1, 0, 0]
        
        images.append(img)
        labels.append(size // 2 - 1)  # 2->0, 4->1, 6->2
    
    return np.array(images), np.array(labels)

def generate_occlusion_data(n=3000):
    """Test occlusion recognition"""
    images = []
    labels = []  # 0=no occlusion, 1=occluded
    
    for _ in range(n):
        x, y = random.uniform(8, 24), random.uniform(8, 24)
        occluded = random.choice([True, False])
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        
        # Draw ball
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx*dx + dy*dy <= 9:
                    px, py = int(x+dx), int(y+dy)
                    if 0 <= px < 32 and 0 <= py < 32:
                        # Occlude part of it
                        if not (occluded and px > x and py < y):
                            img[py, px] = [1, 0, 0]
        
        images.append(img)
        labels.append(1 if occluded else 0)
    
    return np.array(images), np.array(labels)

# Scale test
print("\n1. Scale invariance test...")
X_scale, Y_scale = generate_scale_data(3000)
X_scale = torch.FloatTensor(X_scale).permute(0, 3, 1, 2)
Y_scale = torch.LongTensor(Y_scale)

print("2. Occlusion test...")
X_occ, Y_occ = generate_occlusion_data(3000)
X_occ = torch.FloatTensor(X_occ).permute(0, 3, 1, 2)
Y_occ = torch.LongTensor(Y_occ)

# Model
class CNN(nn.Module):
    def __init__(self, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Scale test
print("\n3. Training scale classifier...")
m = CNN(3)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(10):
    idx = torch.randperm(len(X_scale))
    for i in range(0, len(X_scale), 32):
        p = m(X_scale[idx[i:i+32]])
        loss = F.cross_entropy(p, Y_scale[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    pred = m(X_scale).argmax(dim=1)
    scale_acc = (pred == Y_scale).float().mean().item()

print(f"   Scale accuracy: {scale_acc:.2%}")

# Occlusion test
print("\n4. Training occlusion classifier...")
m = CNN(2)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(10):
    idx = torch.randperm(len(X_occ))
    for i in range(0, len(X_occ), 32):
        p = m(X_occ[idx[i:i+32]])
        loss = F.cross_entropy(p, Y_occ[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    pred = m(X_occ).argmax(dim=1)
    occ_acc = (pred == Y_occ).float().mean().item()

print(f"   Occlusion accuracy: {occ_acc:.2%}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Scale invariance: {scale_acc:.2%}")
print(f"Occlusion recognition: {occ_acc:.2%}")
