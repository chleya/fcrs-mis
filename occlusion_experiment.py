"""
OCCLUSION EXPERIMENT - N Objects with Occlusion

Goal: Test if Slot becomes necessary when objects occlude each other
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
print("OCCLUSION EXPERIMENT")
print("="*60)

def generate_data(n_objects, n_samples=1000, occlusion_prob=0.5):
    """Generate data with N objects and occlusion"""
    X0, X10, targets = [], [], []
    
    for _ in range(n_samples):
        positions = []
        velocities = []
        
        # Generate objects with different depths (z)
        for i in range(n_objects):
            x = random.uniform(5, 27)
            y = random.uniform(8, 24)
            z = i / (n_objects - 1) if n_objects > 1 else 0.5  # Depth 0-1
            positions.append((x, y, z))
            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)
            velocities.append((vx, vy, 0))
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        for i in range(n_objects):
            x, y, z = positions[i]
            if i == 0:
                img0[int(y), int(x)] = [1, 0, 0]  # Red = target
            elif i == 1:
                img0[int(y), int(x)] = [0, 0, 1]  # Blue
            else:
                img0[int(y), int(x)] = [0.5, 0.5, 0.5]
        
        # Sort by depth for occlusion
        sorted_idx = sorted(range(n_objects), key=lambda i: positions[i][2], reverse=True)
        
        # Draw with occlusion
        img10 = np.zeros((32, 32, 3), np.float32)
        final_positions = []
        
        for step in range(10):
            for i in range(n_objects):
                x, y, z = positions[i]
                vx, vy, _ = velocities[i]
                x += vx * 0.5
                y += vy * 0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i] = (x, y, z)
                velocities[i] = (vx, vy, 0)
        
        # Render with occlusion (draw in depth order)
        rendered = np.zeros((32, 32, 3), np.float32)
        for i in sorted_idx:
            x, y, z = positions[i]
            px, py = int(x), int(y)
            if 0 <= px < 32 and 0 <= py < 32:
                if step == 9:  # Final frame
                    # Occlusion: if another object is closer, don't draw
                    is_occluded = False
                    for j in sorted_idx:
                        if j == i: break
                        x2, y2, z2 = positions[j]
                        if abs(x - x2) < 2 and abs(y - y2) < 2 and z2 > z:
                            is_occluded = True
                            break
                    
                    if not is_occluded:
                        rendered[py, px] = [1, 1, 1]
        
        X0.append(img0)
        X10.append(rendered)
        # Target: position of object 0 (closest/red)
        # Sort by depth - object 0 is closest
        closest_idx = sorted_idx[-1]  # Closest
        targets.append(positions[closest_idx][0] / 32)
    
    return np.array(X0), np.array(X10), np.array(targets)

class Baseline(nn.Module):
    def __init__(self, n_objects):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8*2, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x0, x10):
        h0 = self.enc(x0).flatten(1)
        h10 = self.enc(x10).flatten(1)
        return self.fc(torch.cat([h0, h10], dim=1)).squeeze()

class SlotModel(nn.Module):
    def __init__(self, n_objects):
        super().__init__()
        self.n_objects = n_objects
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slots = nn.Parameter(torch.randn(n_objects, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x0, x10):
        h = self.enc(x10).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

print("\n" + "="*60)
print("RESULTS - WITH OCCLUSION")
print("="*60)

results = []

for n in [2, 4, 6]:
    print(f"\n--- N = {n} objects with occlusion ---")
    
    X0, X10, Y = generate_data(n, 1000)
    X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
    X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
    Y = torch.FloatTensor(Y)
    
    random_mse = Y.var().item()
    
    # Baseline
    m = Baseline(n)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(8):
        idx = torch.randperm(len(X0))
        for i in range(0, len(X0), 64):
            p = m(X0[idx[i:i+64]], X10[idx[i:i+64]])
            loss = F.mse_loss(p, Y[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse_b = F.mse_loss(m(X0, X10), Y).item()
    
    # Slot
    m = SlotModel(n)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(8):
        idx = torch.randperm(len(X0))
        for i in range(0, len(X0), 64):
            p = m(X0[idx[i:i+64]], X10[idx[i:i+64]])
            loss = F.mse_loss(p, Y[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse_s = F.mse_loss(m(X0, X10), Y).item()
    
    print(f"Baseline: MSE={mse_b:.4f} ({(random_mse-mse_b)/random_mse*100:.1f}% < random)")
    print(f"Slot:    MSE={mse_s:.4f} ({(random_mse-mse_s)/random_mse*100:.1f}% < random)")
    
    results.append({'n': n, 'baseline': mse_b, 'slot': mse_s})

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for r in results:
    winner = "Baseline" if r['baseline'] < r['slot'] else "Slot"
    print(f"N={r['n']}: Baseline={r['baseline']:.4f}, Slot={r['slot']:.4f} -> {winner}")
