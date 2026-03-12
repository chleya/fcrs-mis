"""Identity Test + Occlusion - Key Experiment"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dataset with OCCLUSION
print("Creating dataset with OCCLUSION...")

data = []
for i in range(6000):
    # 1/3: normal crossing, 1/3: occlusion, 1/3: non-crossing
    mode = i % 3
    
    if mode == 0:  # Normal crossing
        t = (i // 3) * 0.01
        b1_x, b1_y = 5.0 + t * 100, 16.0
        b2_x, b2_y = 27.0 - t * 100, 16.0
        occl = 0
    elif mode == 1:  # Occlusion - ball hides behind wall
        t = (i // 3) * 0.01
        b1_x, b1_y = 5.0 + t * 100, 16.0
        b2_x, b2_y = 15.0, 16.0  # Hidden behind wall at x=15
        occl = 1
    else:  # Non-crossing
        t = (i // 3) * 0.002
        b1_x, b1_y = 10.0, 10.0 + t * 50
        b2_x, b2_y = 22.0, 22.0 + t * 50
        occl = 0
    
    # Clamp
    b1_x = min(31, max(0, int(b1_x)))
    b1_y = min(31, max(0, int(b1_y)))
    b2_x = min(31, max(0, int(b2_x)))
    b2_y = min(31, max(0, int(b2_y)))
    
    # Image
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[b1_y, b1_x] = [1, 1, 1]  # Ball 1 - white
    if mode != 1:
        img[b2_y, b2_x] = [0.5, 0.5, 0.5]  # Ball 2 - gray (visible)
    
    # Add wall for occlusion
    if mode == 1:
        img[10:22, 15] = [0.8, 0.2, 0.2]  # Red wall
    
    # Target: ball1 position at t+3
    if mode == 0:
        target = [(5.0 + (t*100+3)) / 32, 16.0 / 32]
    elif mode == 1:
        target = [(5.0 + (t*100+3)) / 32, 16.0 / 32]
    else:
        target = [10.0 / 32, (10.0 + t * 50 + 3) / 32]
    
    data.append((img, target, occl, mode))

images = np.array([d[0] for d in data])
targets = np.array([d[1] for d in data])
occlusion = np.array([d[2] for d in data])
mode_arr = np.array([d[3] for d in data])

X = torch.FloatTensor(images).permute(0, 3, 1, 2)
Y = torch.FloatTensor(targets)

print(f"Dataset: {len(X)} samples")
print(f"  Normal crossing: {(mode_arr==0).sum()}")
print(f"  Occlusion: {(mode_arr==1).sum()}")
print(f"  Non-crossing: {(mode_arr==2).sum()}")

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
        # Two slots
        self.s1 = nn.Sequential(nn.Linear(32*8*8, 32), nn.ReLU(), nn.Linear(32, 2))
        self.s2 = nn.Sequential(nn.Linear(32*8*8, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        # Use slot 0 for prediction
        return self.s1(h), h

print("\nTraining models...")

results = {}
for name, model_fn in [('Baseline', Baseline), ('Slot', SlotModel)]:
    print(f"\n{name}:")
    m = model_fn()
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
            'occl': F.mse_loss(p[occlusion==1], Y[occlusion==1]).item(),
            'normal': F.mse_loss(p[occlusion==0], Y[occlusion==0]).item(),
            'cross': F.mse_loss(p[mode_arr==0], Y[mode_arr==0]).item()
        }
        print(f"  MSE(all): {results[name]['all']:.4f}")
        print(f"  MSE(occlusion): {results[name]['occl']:.4f}")
        print(f"  MSE(normal): {results[name]['normal']:.4f}")

print("\n" + "="*50)
print("SUMMARY - Occlusion Test")
print("="*50)
for name, r in results.items():
    print(f"{name}: occl={r['occl']:.4f}, normal={r['normal']:.4f}, cross={r['cross']:.4f}")
