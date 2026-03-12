"""
SEMANTIC IDENTITY TEST - Position Swap

Key idea: 
- t0: mark A (red), B (blue) at positions
- SWAP positions: A goes to B's position, B goes to A's position
- t1+: white balls, model must track "the ball that was red" not "the left ball"

This forces SEMANTIC identity, not position identity.
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
print("SEMANTIC IDENTITY TEST - POSITION SWAP")
print("="*60)

def generate_data(n=2000):
    """Generate data with position swap"""
    X_t0, X_t5, X_t10 = [], [], []
    Y_a = []  # Position of ball A (the one that was RED at t0)
    
    for _ in range(n):
        # Initial positions
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        # t0: colored markers
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]  # Red = A
        img0[int(y), int(x_b)] = [0, 0, 1]  # Blue = B
        
        # CRITICAL: Swap positions!
        # Now A (red) is at B's original position
        # And B (blue) is at A's original position
        x_a, x_b = x_b, x_a
        
        # Initial velocities (after swap)
        vx_a = random.uniform(-2, 2)
        vx_b = random.uniform(-2, 2)
        
        white = [1, 1, 1]
        
        # t1-t5: move
        for _ in range(1, 6):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
            if x_b < 3 or x_b > 29: vx_b *= -1
            x_a = max(3, min(28, x_a))
            x_b = max(3, min(28, x_b))
        
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a)] = white
        img5[int(y), int(x_b)] = white
        
        # t6-t10
        for _ in range(6, 11):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
            if x_b < 3 or x_b > 29: vx_b *= -1
            x_a = max(3, min(28, x_a))
            x_b = max(3, min(28, x_b))
        
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x_a)] = white
        img10[int(y), int(x_b)] = white
        
        X_t0.append(img0)
        X_t5.append(img5)
        X_t10.append(img10)
        Y_a.append(x_a / 32)  # Position of ball A (semantic identity)
    
    return np.array(X_t0), np.array(X_t5), np.array(X_t10), np.array(Y_a)

print("\n1. Generating data with position swap...")
X0, X5, X10, Y = generate_data(2000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   Data: {X0.shape}")

# Verify: what's the correlation between t0 red position and target?
# If using position (not semantic), this should be near 0 after swap
t0_red_pos = X0[:, 0, 16, :].argmax(axis=1).float().numpy()
corr = np.corrcoef(t0_red_pos, Y.numpy())[0, 1]
print(f"\n   Correlation (t0 red pixel → target): {corr:.3f}")
print(f"   (If ~0: position-based solution fails)")

# Baseline
class Model(nn.Module):
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

print("\n2. Training...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(15):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse = F.mse_loss(m(X0, X5, X10), Y).item()
random_mse = Y.var().item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"MSE: {mse:.4f} ({(random_mse-mse)/random_mse*100:.1f}% < random)")
print(f"Random MSE: {random_mse:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse > random_mse * 0.7:
    print("=> SEMANTIC IDENTITY REQUIRED!")
    print("=> Position-based solution fails (correlation ~0)")
    print("=> Only semantic identity can solve this task")
else:
    print("=> Model found some other pattern")
