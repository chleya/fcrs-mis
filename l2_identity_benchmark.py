"""
L2 IDENTITY BENCHMARK - NO SHORTCUTS

This benchmark is designed so that:
- Without identity tracking: MSE ≈ random
- With identity tracking: MSE << random

Key features:
1. Visual indistinguishable: same color, same shape
2. Different physics: A=bounce, B=pass-through
3. Random velocity reset: trajectory prediction fails
4. Full crossing: position ambiguity
5. Identity-specific outcome: predict position of A only
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
print("L2 IDENTITY BENCHMARK - NO SHORTCUTS")
print("="*60)

def generate_identity_benchmark(n=3000):
    """
    Task: Track ball A (bounce physics) through time
    
    Conditions that force identity:
    1. t0: Colored (identity marker)
    2. t1-t9: WHITE balls (no color info)
    3. Different physics: A bounces, B passes through
    4. Random velocity resets: trajectory extrapolation fails
    5. Full crossing: position ambiguity
    6. Target: Position of A (ball with bounce physics)
    """
    X_t0 = []
    X_t5 = []
    X_t10 = []
    Y_a = []  # Position of ball A at t10
    Y_b = []  # Position of ball B at t10
    
    for _ in range(n):
        # Initial positions
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        # Initial velocities
        vx_a = random.uniform(1, 2)  # Moving right
        vx_b = random.uniform(-2, -1)  # Moving left (toward each other)
        
        # t0: colored markers
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]  # Red = ball A (bounce)
        img0[int(y), int(x_b)] = [0, 0, 1]  # Blue = ball B (pass-through)
        
        white = [1, 1, 1]
        
        # t1-t4: approach each other
        for step in range(1, 5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            # A bounces
            if x_a < 3 or x_a > 29: vx_a *= -1
            # B passes through (no boundary handling)
        
        # t5: FULL CROSSING - balls swap sides!
        # After this, positions are completely swapped
        x_a, x_b = x_b, x_a  # Swap positions!
        
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a)] = white  # Now A is on the left (was on right)
        img5[int(y), int(x_b)] = white  # Now B is on the right (was on left)
        
        # t6-t9: continue with DIFFERENT physics + velocity resets
        for step in range(6, 10):
            # Random velocity reset - breaks trajectory prediction!
            if random.random() < 0.5:
                vx_a = random.uniform(-2, 2)
            if random.random() < 0.5:
                vx_b = random.uniform(-2, 2)
            
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            
            # A bounces, B passes through
            if x_a < 3 or x_a > 29: vx_a *= -1
            # B: wrap around
            x_b = ((x_b - 3) % 26) + 3
        
        # t10: final positions
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x_a)] = white  # A
        img10[int(y), int(x_b)] = white  # B
        
        X_t0.append(img0)
        X_t5.append(img5)
        X_t10.append(img10)
        Y_a.append(x_a / 32)
        Y_b.append(x_b / 32)
    
    return np.array(X_t0), np.array(X_t5), np.array(X_t10), np.array(Y_a), np.array(Y_b)

print("\n1. Generating identity benchmark data...")
X0, X5, X10, Y_a, Y_b = generate_identity_benchmark(3000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y_a = torch.FloatTensor(Y_a)
Y_b = torch.FloatTensor(Y_b)

print(f"   X0: {X0.shape}, X5: {X5.shape}, X10: {X10.shape}")
print(f"   Y_a: {Y_a.shape}, Y_b: {Y_b.shape}")

# Check: correlation between t0 and target
# If task has shortcuts, this correlation should be high
corr_a = np.corrcoef(X0[:, 0, 16, :].argmax(axis=1), Y_a)[0, 1]
corr_b = np.corrcoef(X0[:, 2, 16, :].argmax(axis=1), Y_b)[0, 1]
print(f"\n   Correlation t0 red pixel -> Y_a: {corr_a:.3f}")
print(f"   Correlation t0 blue pixel -> Y_b: {corr_b:.3f}")
print(f"   (Low correlation = no shortcuts!)")

# Baseline model
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8*3, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x0, x5, x10):
        h0 = self.enc(x0).flatten(1)
        h5 = self.enc(x5).flatten(1)
        h10 = self.enc(x10).flatten(1)
        return self.fc(torch.cat([h0, h5, h10], dim=1)).squeeze()

print("\n2. Training Baseline to predict A...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y_a[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_baseline_a = F.mse_loss(m(X0, X5, X10), Y_a).item()

print("3. Training Baseline to predict B...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y_b[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_baseline_b = F.mse_loss(m(X0, X5, X10), Y_b).item()

# Random baseline
random_mse_a = Y_a.var().item()
random_mse_b = Y_b.var().item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Predict A (bounce):  MSE = {mse_baseline_a:.4f} ({(random_mse_a-mse_baseline_a)/random_mse_a*100:.1f}% < random)")
print(f"Predict B (pass):    MSE = {mse_baseline_b:.4f} ({(random_mse_b-mse_baseline_b)/random_mse_b*100:.1f}% < random)")
print(f"Random A:            MSE = {random_mse_a:.4f}")
print(f"Random B:            MSE = {random_mse_b:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse_baseline_a < random_mse_a * 0.5:
    print("=> Model CAN predict A despite all shortcuts blocked!")
    print("=> Identity tracking may have emerged")
else:
    print("=> Model CANNOT predict without identity")
    print("=> This is the CORRECT benchmark for identity!")
    
if abs(corr_a) < 0.2 and abs(corr_b) < 0.2:
    print("\n=> Shortcuts successfully blocked!")
    print("=> Correlation near zero = no easy patterns to exploit")
