"""
STEP 1: Attribute-Position Binding Test
Task: Input 2 balls (red left blue right OR red right blue left randomly)
Output: Classify "Is red ball on the left?"

This task REQUIRES binding - knowing "red" AND "left" together is useless
without knowing they belong to the SAME object.
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
print("ATTRIBUTE-POSITION BINDING TEST")
print("="*60)

def generate_binding_data(n=2000):
    """Generate red/blue balls at random positions"""
    images = []
    labels = []  # 1 if red is on left, 0 if red is on right
    
    for _ in range(n):
        # Random positions for both balls
        pos1 = random.uniform(0.25, 0.75)  # position 1
        pos2 = random.uniform(0.25, 0.75)  # position 2
        
        # Random assignment: which position gets red?
        red_left = random.choice([True, False])
        
        x1 = int(pos1 * 32)
        x2 = int(pos2 * 32)
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        
        # Ball at position 1
        if red_left:
            img[16, x1] = [1, 0, 0]  # Red
            img[16, x2] = [0, 0, 1]  # Blue
            label = 1 if x1 < x2 else 0
        else:
            img[16, x1] = [0, 0, 1]  # Blue
            img[16, x2] = [1, 0, 0]  # Red
            label = 0 if x1 < x2 else 1
        
        images.append(img)
        labels.append(label)
    
    return np.array(images), np.array(labels)

print("\n1. Generating binding data...")
X, Y = generate_binding_data(2000)
X = torch.FloatTensor(X).permute(0, 3, 1, 2)
Y = torch.LongTensor(Y)

print(f"   X: {X.shape}, Y: {Y.shape}")
print(f"   Label distribution: {Y.sum()}/{len(Y)} (red on left)")

# Model: CNN baseline
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

print("\n2. Training baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(15):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p = m(X[idx[i:i+32]])
        loss = F.cross_entropy(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    pred = m(X).argmax(dim=1)
    acc = (pred == Y).float().mean().item()

print(f"\n3. RESULTS:")
print(f"   Baseline accuracy: {acc:.1%}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

random_acc = 0.5
if acc < random_acc + 0.15:
    print(f"~{acc:.0%} ≈ {random_acc:.0%} (random guess)")
    print("=> NO binding capability!")
    print("   Model cannot bind 'red' + 'left' to same object")
else:
    print(f"{acc:.0%} > {random_acc:.0%}")
    print("=> HAS binding capability!")

print(f"\nIf this is ~50%, it confirms:")
print("1. Model encodes 'red' separately from 'left'")
print("2. No object-position binding exists")
print("3. This is why identity tracking fails")
