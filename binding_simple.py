"""
Object-Position Binding Test - Simplified
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
print("OBJECT-POSITION BINDING TEST")
print("="*60)

# Generate data
def generate_data(n=2000):
    images = []
    pos_labels = []
    obj_labels = []
    
    for i in range(n):
        # Random position (0.2-0.8)
        p = random.uniform(0.2, 0.8)
        x, y = int(p * 32), int(p * 32)
        
        # Random object (0=red, 1=blue)
        obj = i % 2
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        if obj == 0:
            img[y, x] = [1, 0, 0]
        else:
            img[y, x] = [0, 0, 1]
        
        images.append(img)
        pos_labels.append(p)
        obj_labels.append(obj)
    
    return np.array(images), np.array(pos_labels), np.array(obj_labels)

print("\n1. Generating data...")
X, P, O = generate_data(2000)
X = torch.FloatTensor(X).permute(0, 3, 1, 2)
P = torch.FloatTensor(P)
O = torch.LongTensor(O)

# Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc_pos = nn.Linear(64*8*8, 1)
        self.fc_obj = nn.Linear(64*8*8, 2)
        
    def forward(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_pos(h), self.fc_obj(h)

print("\n2. Training...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(8):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        pos_p, obj_p = m(X[idx[i:i+32]])
        loss = F.mse_loss(pos_p.squeeze(), P[idx[i:i+32]]) + F.cross_entropy(obj_p, O[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()

print("\n3. Analyzing...")

with torch.no_grad():
    # Get latent representations
    h = m.enc(X).flatten(1)
    
    # Split by object
    h_red = h[O == 0].mean(axis=0)
    h_blue = h[O == 1].mean(axis=0)
    
    # Split by position
    h_left = h[P < 0.5].mean(axis=0)
    h_right = h[P >= 0.5].mean(axis=0)
    
    # Correlation analysis
    obj_diff = (h_red - h_blue).abs().mean().item()
    pos_diff = (h_left - h_right).abs().mean().item()
    
    print(f"\nRepresentation analysis:")
    print(f"   Object difference (red vs blue): {obj_diff:.4f}")
    print(f"   Position difference (left vs right): {pos_diff:.4f}")
    
    # Check what's dominant
    if pos_diff > obj_diff * 1.5:
        print("\n=> Position is DOMINANT in representation")
        print("   Model encodes position regardless of object")
    elif obj_diff > pos_diff * 1.5:
        print("\n=> Object is DOMINANT in representation")  
        print("   Model encodes object regardless of position")
    else:
        print("\n=> Both are encoded")
        
    # Correlation test
    red_left = h[(O==0) & (P<0.5)]
    red_right = h[(O==0) & (P>=0.5)]
    blue_left = h[(O==1) & (P<0.5)]
    blue_right = h[(O==1) & (P>=0.5)]
    
    corr_left = np.corrcoef(red_left.mean(axis=0), blue_left.mean(axis=0))[0,1]
    corr_right = np.corrcoef(red_right.mean(axis=0), blue_right.mean(axis=0))[0,1]
    
    print(f"\n   Left position: red-blue correlation: {corr_left:.4f}")
    print(f"   Right position: red-blue correlation: {corr_right:.4f}")
    
    if abs(corr_left) > 0.9 and abs(corr_right) > 0.9:
        print("\n=> Object binding is WEAK")
        print("   Same object at different positions has similar representation")
    else:
        print("\n=> Object-position binding EXISTS!")
        print("   Different positions affect object representation")
