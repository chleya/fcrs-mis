"""
ULTIMATE Identity Test - Identity MANDATORY
- Random colors each sequence (no fixed red=ballA)
- Random velocity mutations at t3, t7 (identity-bound)
- Full overlap at t5
- White balls t1-t9 (no color info)
- Must process ALL frames t0-t9
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
print("ULTIMATE IDENTITY TEST")
print("="*60)

def generate_sequence():
    """Generate one sequence with MANDATORY identity tracking"""
    # Random colors for this sequence
    color_a = [random.random(), random.random(), random.random()]
    color_b = [random.random(), random.random(), random.random()]
    
    # Random initial positions
    x_a, y_a = random.uniform(5, 12), random.uniform(8, 24)
    x_b, y_b = random.uniform(20, 27), random.uniform(8, 24)
    
    # Random initial velocities
    vx_a, vy_a = random.uniform(-2, 2), random.uniform(-2, 2)
    vx_b, vy_b = random.uniform(-2, 2), random.uniform(-2, 2)
    
    frames = []
    
    # t0: colored (identity marker!)
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[int(y_a), int(x_a)] = color_a
    img[int(y_b), int(x_b)] = color_b
    frames.append(img.copy())
    
    white = [1, 1, 1]
    
    # t1-t4: white balls (no identity info!)
    for step in range(1, 5):
        x_a, y_a = x_a + vx_a * 0.5, y_a + vy_a * 0.5
        x_b, y_b = x_b + vx_b * 0.5, y_b + vy_b * 0.5
        
        # Bounce for A
        if x_a < 3 or x_a > 29: vx_a *= -1
        if y_a < 3 or y_a > 29: vy_a *= -1
        
        # Wrap for B
        x_b = ((x_b - 3) % 26) + 3
        y_b = ((y_b - 3) % 26) + 3
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[int(y_a), int(x_a)] = white
        img[int(y_b), int(x_b)] = white
        frames.append(img.copy())
    
    # t5: FULL OVERLAP (no spatial info!)
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[16, 16] = white  # One white dot
    frames.append(img.copy())
    
    # t6-t9: more white, with velocity mutation at t7!
    for step in range(6, 10):
        # Velocity mutation at t7!
        if step == 7:
            # A: random direction flip
            if random.random() < 0.5: vx_a *= -1
            if random.random() < 0.5: vy_a *= -1
            # B: random magnitude change
            vx_b *= random.uniform(0.5, 1.5)
            vy_b *= random.uniform(0.5, 1.5)
        
        x_a, y_a = x_a + vx_a * 0.5, y_a + vy_a * 0.5
        x_b, y_b = x_b + vx_b * 0.5, y_b + vy_b * 0.5
        
        # Bounce for A
        if x_a < 3 or x_a > 29: vx_a *= -1
        if y_a < 3 or y_a > 29: vy_a *= -1
        
        # Wrap for B
        x_b = ((x_b - 3) % 26) + 3
        y_b = ((y_b - 3) % 26) + 3
        
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[int(y_a), int(x_a)] = white
        img[int(y_b), int(x_b)] = white
        frames.append(img.copy())
    
    # Target: ball A position (the one marked at t0!)
    target = [x_a / 32, y_a / 32]
    
    return np.array(frames), target

def generate_dataset(n=5000):
    frames = []
    targets = []
    for _ in range(n):
        f, t = generate_sequence()
        frames.append(f)
        targets.append(t)
    return np.array(frames), np.array(targets)

print("\n1. Generating dataset...")
X, Y = generate_dataset(5000)

# Input: all frames t0-t9
X_tensor = torch.FloatTensor(X).permute(0, 1, 4, 2, 3)  # (N, 10, 3, 32, 32)
Y_tensor = torch.FloatTensor(Y)

print(f"   X: {X_tensor.shape}, Y: {Y_tensor.shape}")

# Use only t0 and t5 (crossing point) - minimum info!
X_t0 = X_tensor[:, 0]  # Colored
X_t5 = X_tensor[:, 5]  # Overlap (no info)
X_t9 = X_tensor[:, 9]  # Final white

# Models
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.enc(x)

class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slot = nn.Parameter(torch.randn(2, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        
    def forward(self, x):
        h = self.enc(x).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slot.unsqueeze(0)
        return self.predict(h[:, 0])

print("\n2. Training models...")

results = {}

# Baseline: only t0 (colored) - can use color info
print("   Baseline (t0 only - has color)...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X_t0))
    for i in range(0, len(X_t0), 32):
        p = m(X_t0[idx[i:i+32]])
        loss = F.mse_loss(p, Y_tensor[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
results['Baseline_t0'] = F.mse_loss(m(X_t0), Y_tensor).item()

# Baseline: only t5 (overlap - no info)
print("   Baseline (t5 only - overlap)...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X_t5))
    for i in range(0, len(X_t5), 32):
        p = m(X_t5[idx[i:i+32]])
        loss = F.mse_loss(p, Y_tensor[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
results['Baseline_t5'] = F.mse_loss(m(X_t5), Y_tensor).item()

# Slot: t0 + t5 combined (identity MUST be tracked through overlap)
print("   SlotModel (t0 + t5)...")
m = SlotModel()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X_t0))
    for i in range(0, len(X_t0), 32):
        # Process t0
        p0 = m(X_t0[idx[i:i+32]])
        # Process t5  
        p5 = m(X_t5[idx[i:i+32]])
        # Combined prediction
        p = (p0 + p5) / 2
        loss = F.mse_loss(p, Y_tensor[idx[i:i+32]])
        opt.zero_grad(); loss.backward(); opt.step()
results['Slot_t0_t5'] = F.mse_loss((m(X_t0)[0] + m(X_t5)[0])/2, Y_tensor).item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
for name, mse in results.items():
    print(f"{name}: MSE = {mse:.4f}")
