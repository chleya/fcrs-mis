"""
Object-Position Binding Test
Does model form "object at position" binding, or just separate representations?
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

# Test 1: Same position, different objects
def generate_same_pos_diff_obj(n=1000):
    """Same (x,y), different objects"""
    images = []
    positions = []
    objects = []  # 0=red, 1=blue
    
    for i in range(n):
        pos = random.uniform(0.2, 0.8)  # Same position range
        x, y = pos * 32, pos * 32
        obj = i % 2  # Alternate red/blue
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        if obj == 0:
            img[int(y), int(x)] = [1, 0, 0]  # Red
        else:
            img[int(y), int(x)] = [0, 0, 1]  # Blue
        
        images.append(img)
        positions.append([pos, pos])
        objects.append(obj)
    
    return np.array(images), np.array(positions), np.array(objects)

# Test 2: Different positions, same object
def generate_diff_pos_same_obj(n=1000):
    """Different positions, same object"""
    images = []
    positions = []
    objects = []
    
    for i in range(n):
        pos = random.uniform(0.2, 0.8)
        x, y = pos * 32, pos * 32
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[int(y), int(x)] = [1, 0, 0]  # Always red
        
        images.append(img)
        positions.append([pos, pos])
        objects.append(0)
    
    return np.array(images), np.array(positions), np.array(objects)

# Test 3: Both vary
def generate_both_vary(n=1000):
    images = []
    positions = []
    objects = []
    
    for i in range(n):
        pos = random.uniform(0.2, 0.8)
        x, y = pos * 32, pos * 32
        obj = i % 2
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        if obj == 0:
            img[int(y), int(x)] = [1, 0, 0]
        else:
            img[int(y), int(x)] = [0, 0, 1]
        
        images.append(img)
        positions.append([pos, pos])
        objects.append(obj)
    
    return np.array(images), np.array(positions), np.array(objects)

# Generate data
print("\n1. Generating data...")
X1, P1, O1 = generate_same_pos_diff_obj(1000)  # Same pos, diff obj
X2, P2, O2 = generate_diff_pos_same_obj(1000)  # Diff pos, same obj
X3, P3, O3 = generate_both_vary(1000)  # Both vary

X1 = torch.FloatTensor(X1).permute(0, 3, 1, 2)
X2 = torch.FloatTensor(X2).permute(0, 3, 1, 2)
X3 = torch.FloatTensor(X3).permute(0, 3, 1, 2)

print(f"   Same pos/diff obj: {X1.shape}")
print(f"   Diff pos/same obj: {X2.shape}")
print(f"   Both vary: {X3.shape}")

# Model with latent extraction
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
        )
        self.fc_pos = nn.Linear(128*4*4, 2)
        self.fc_obj = nn.Linear(128*4*64, 2)  # Object classifier
        
    def forward(self, x):
        h = self.encoder(x).flatten(1)
        pos_pred = self.fc_pos(h)
        
        # Reshape for object
        h2 = self.encoder(x).flatten(2)
        obj_pred = self.fc_obj(h2.mean(dim=2))
        
        return pos_pred, obj_pred, h

print("\n2. Training model...")

# Train on "both vary" data
m = CNN()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

X3_pos = torch.FloatTensor(P3)
X3_obj = torch.LongTensor(O3)

for ep in range(10):
    idx = torch.randperm(len(X3))
    for i in range(0, len(X3), 32):
        pos_p, obj_p, _ = m(X3[idx[i:i+32]])
        loss_pos = F.mse_loss(pos_p, X3_pos[idx[i:i+32]])
        loss_obj = F.cross_entropy(obj_p, X3_obj[idx[i:i+32]])
        loss = loss_pos + loss_obj
        opt.zero_grad(); loss.backward(); opt.step()

print("\n3. Analyzing representations...")

# Extract latent from different conditions
with torch.no_grad():
    _, _, h1 = m(X1)  # Same pos, diff obj
    _, _, h2 = m(X2)  # Diff pos, same obj
    _, _, h3 = m(X3)  # Both vary

h1 = h1.flatten(1).numpy()
h2 = h2.flatten(1).numpy()
h3 = h3.flatten(1).numpy()

# Analysis 1: Position encoding independence
# If position is encoded independently, same position should have similar latent regardless of object
pos_similarity = np.corrcoef(h1[:100].mean(axis=0), h1[100:].mean(axis=0))[0,1]
print(f"\nSame position, different objects:")
print(f"   Latent correlation: {pos_similarity:.4f}")
if abs(pos_similarity) > 0.9:
    print("   -> Position encoded INDEPENDENTLY of object")

# Analysis 2: Object encoding independence  
# If object is encoded independently, same object should have similar latent regardless of position
obj_similarity = np.corrcoef(h2[:100].mean(axis=0), h2[100:].mean(axis=0))[0,1]
print(f"\nDifferent position, same object:")
print(f"   Latent correlation: {obj_similarity:.4f}")
if abs(obj_similarity) > 0.9:
    print("   -> Object encoded INDEPENDENTLY of position")

# Analysis 3: Binding test
# If "object at position" is bound, changing both should change latent most
diff_both = np.linalg.norm(h3[500:] - h3[:500], axis=1).mean()
diff_pos = np.linalg.norm(h2[500:] - h2[:500], axis=1).mean()
diff_obj = np.linalg.norm(h1[500:] - h1[:500], axis=1).mean()

print(f"\nRepresentation change analysis:")
print(f"   Both vary: {diff_both:.2f}")
print(f"   Position varies: {diff_pos:.2f}")  
print(f"   Object varies: {diff_obj:.2f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if diff_pos > diff_obj * 1.5:
    print("-> Model is MORE sensitive to position")
    print("-> Position may be dominant in representation")
if diff_obj > diff_pos * 1.5:
    print("-> Model is MORE sensitive to object")
    print("-> Object may be dominant in representation")
