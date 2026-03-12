"""
N OBJECTS SCALING EXPERIMENT

Goal: Find critical N where slot becomes necessary

Hypothesis:
- N=1: Baseline best (no objects to factorize)
- N=2: Baseline ≈ Slot (simple case)
- N=3: Baseline starts to struggle
- N>=4: Slot advantage emerges
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
print("N OBJECTS SCALING EXPERIMENT")
print("="*60)

def generate_data(n_objects, n_samples=1000):
    """Generate data with N objects"""
    X0, X10, targets = [], [], []
    
    for _ in range(n_samples):
        # Generate N objects
        positions = []
        velocities = []
        colors = []
        
        for i in range(n_objects):
            # Random position
            x = random.uniform(5, 27)
            y = random.uniform(8, 24)
            positions.append((x, y))
            
            # Random velocity
            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)
            velocities.append((vx, vy))
            
            # Color (t0 only)
            if i == 0:
                colors.append([1, 0, 0])  # Red = target
            elif i == 1:
                colors.append([0, 0, 1])  # Blue
            else:
                colors.append([random.random(), random.random(), random.random()])
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        for i in range(n_objects):
            x, y = positions[i]
            img0[int(y), int(x)] = colors[i]
        
        # Simulate to t10
        for step in range(10):
            for i in range(n_objects):
                x, y = positions[i]
                vx, vy = velocities[i]
                x += vx * 0.5
                y += vy * 0.5
                
                # Bounce
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                
                positions[i] = (x, y)
                velocities[i] = (vx, vy)
        
        # t10: white
        img10 = np.zeros((32, 32, 3), np.float32)
        final_positions = []
        for i in range(n_objects):
            x, y = positions[i]
            img10[int(y), int(x)] = [1, 1, 1]
            final_positions.append(x / 32)
        
        X0.append(img0)
        X10.append(img10)
        # Target: position of object 0 (red at t0)
        targets.append(final_positions[0])
    
    return np.array(X0), np.array(X10), np.array(targets)

# Models
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
        h = self.enc(x10).mean(dim=[2, 3])  # (B, 64)
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)  # (B, N, 64)
        # Return prediction from slot 0 (target object)
        return self.predict(h[:, 0]).squeeze()

print("\n" + "="*60)
print("RESULTS")
print("="*60)

results = []

for n in [1, 2, 3, 4, 5, 6]:
    print(f"\n--- N = {n} objects ---")
    
    # Generate data
    X0, X10, Y = generate_data(n, 1000)
    X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
    X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
    Y = torch.FloatTensor(Y)
    
    random_mse = Y.var().item()
    
    # Train Baseline
    m = Baseline(n)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(8):
        idx = torch.randperm(len(X0))
        for i in range(0, len(X0), 64):
            p = m(X0[idx[i:i+64]], X10[idx[i:i+64]])
            loss = F.mse_loss(p, Y[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse_baseline = F.mse_loss(m(X0, X10), Y).item()
    
    # Train Slot
    m = SlotModel(n)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(8):
        idx = torch.randperm(len(X0))
        for i in range(0, len(X0), 64):
            p = m(X0[idx[i:i+64]], X10[idx[i:i+64]])
            loss = F.mse_loss(p, Y[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse_slot = F.mse_loss(m(X0, X10), Y).item()
    
    imp_baseline = (random_mse - mse_baseline) / random_mse * 100
    imp_slot = (random_mse - mse_slot) / random_mse * 100
    
    print(f"Baseline: MSE={mse_baseline:.4f} ({imp_baseline:.1f}% < random)")
    print(f"Slot:    MSE={mse_slot:.4f} ({imp_slot:.1f}% < random)")
    
    results.append({
        'n': n,
        'baseline': mse_baseline,
        'slot': mse_slot,
        'random': random_mse
    })

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'N':>3} | {'Baseline':>10} | {'Slot':>10} | {'Winner':>8}")
print("-" * 40)
for r in results:
    winner = "Baseline" if r['baseline'] < r['slot'] else "Slot"
    print(f"{r['n']:>3} | {r['baseline']:>10.4f} | {r['slot']:>10.4f} | {winner:>8}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("If Baseline consistently wins:")
print("=> Task is still too simple")
print("=> Need more complex scenario (occlusion, interaction)")
print("\nIf Slot wins at N>=4:")
print("=> Found critical complexity threshold!")
print("=> Structure becomes necessary at scale")
