"""
Spatial Cognition - Translation Invariance Test
Can model recognize object regardless of position?
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
print("TRANSLATION INVARIANCE TEST")
print("="*60)

def generate_translation_data(n=3000):
    """Test if model can recognize ball at different positions"""
    images = []
    positions = []  # Original position (before translation)
    
    for _ in range(n):
        # Random original position
        orig_x = random.uniform(10, 22)
        orig_y = random.uniform(10, 22)
        
        # Random translation
        tx = random.uniform(-5, 5)
        ty = random.uniform(-5, 5)
        
        # Final position (with translation)
        x = orig_x + tx
        y = orig_y + ty
        x = max(3, min(29, x))
        y = max(3, min(29, y))
        
        # Image with ball at translated position
        img = np.zeros((32, 32, 3), dtype=np.float32)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx*dx + dy*dy <= 4:
                    px, py = int(x+dx), int(y+dy)
                    if 0 <= px < 32 and 0 <= py < 32:
                        img[py, px] = [1, 0, 0]
        
        images.append(img)
        positions.append([orig_x/32, orig_y/32])  # Original, not translated!
    
    return np.array(images), np.array(positions)

print("\n1. Generating translation data...")
X, Y = generate_translation_data(3000)
X = torch.FloatTensor(X).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   X: {X.shape}, Y: {Y.shape}")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*4*4, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

print("\n2. Training...")
m = CNN()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(10):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 32):
        p = m(X[idx[i:i+32]])
        loss = F.mse_loss(p, Y[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    mse = F.mse_loss(m(X), Y).item()

print(f"\n   Translation MSE: {mse:.6f}")

# Also test: does model learn translation or absolute position?
# If MSE is low, model learned translation-invariant representation
# If MSE is high, model learned position-dependent

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse < 0.01:
    print("Model learned TRANSLATION-INVARIANT representation")
    print("(can recognize object regardless of position)")
else:
    print("Model learned position-dependent features")
