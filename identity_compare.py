"""Identity Test - Baseline vs Slot"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dataset
print("Creating crossing dataset...")
data = []
for i in range(6000):
    if i < 3000:  # Crossing
        b1_x, b1_y = 5.0 + i * 0.01, 16.0
        b2_x, b2_y = 27.0 - i * 0.01, 16.0
    else:  # Non-crossing
        b1_x, b1_y = 10.0, 10.0 + (i-3000) * 0.002
        b2_x, b2_y = 22.0, 22.0 + (i-3000) * 0.002
    
    b1_x = min(31, max(0, int(b1_x)))
    b1_y = min(31, max(0, int(b1_y)))
    b2_x = min(31, max(0, int(b2_x)))
    b2_y = min(31, max(0, int(b2_y)))
    
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[b1_y, b1_x] = [1, 1, 1]
    img[b2_y, b2_x] = [1, 1, 1]
    
    if i < 3000:
        target = [(5.0 + (i+3) * 0.01) / 32, 16.0 / 32]
    else:
        target = [10.0 / 32, (10.0 + (i-3000+3) * 0.002) / 32]
    
    data.append((img, target, 1 if i < 3000 else 0))

images = np.array([d[0] for d in data])
targets = np.array([d[1] for d in data])
crossing = np.array([d[2] for d in data])

X = torch.FloatTensor(images).permute(0, 3, 1, 2)
Y = torch.FloatTensor(targets)

# Models
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32*8*8, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        return self.fc(h), h

class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU())
        # Simple slot-like: two parallel pathways
        self.slot1 = nn.Sequential(nn.Linear(32*8*8, 32), nn.ReLU(), nn.Linear(32, 2))
        self.slot2 = nn.Sequential(nn.Linear(32*8*8, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        # Use slot 0 (should track ball1 if object identity learned)
        return self.slot1(h), h

print("\nTraining models...")

results = {}
for name, model_class in [('Baseline', Baseline), ('Slot', SlotModel)]:
    print(f"\n{name}:")
    m = model_class()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    
    for ep in range(10):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            p, _ = m(X[idx[i:i+32]])
            loss = F.mse_loss(p, Y[idx[i:i+32]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    m.eval()
    with torch.no_grad():
        p, _ = m(X)
        results[name] = {
            'all': F.mse_loss(p, Y).item(),
            'cross': F.mse_loss(p[crossing==1], Y[crossing==1]).item(),
            'normal': F.mse_loss(p[crossing==0], Y[crossing==0]).item()
        }
        print(f"  MSE(all): {results[name]['all']:.4f}")
        print(f"  MSE(cross): {results[name]['cross']:.4f}")
        print(f"  MSE(normal): {results[name]['normal']:.4f}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for name, r in results.items():
    print(f"{name}: cross={r['cross']:.4f}, normal={r['normal']:.4f}")
