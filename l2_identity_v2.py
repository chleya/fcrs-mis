"""
L2 IDENTITY BENCHMARK v2 - SAME PHYSICS

Key change from v1:
- Both balls use SAME physics (both bounce)
- ONLY difference is initial position/motion
- This forces identity as the only discriminating factor

Design:
1. t0: Colored markers (A=red, B=blue)
2. t1-t9: WHITE balls, SAME physics (both bounce)
3. Random velocity resets: break trajectory prediction
4. Full crossing: position ambiguity
5. Predict: position of A

If identity is needed:
- Model must track which ball was A
- Without identity: cannot predict A's specific position
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
print("L2 IDENTITY BENCHMARK v2 - SAME PHYSICS")
print("="*60)

def generate_data(n=3000):
    X_t0, X_t5, X_t10 = [], [], []
    Y_a = []
    
    for _ in range(n):
        # Same initial setup
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        # Same physics: both bounce
        vx_a = random.uniform(-2, 2)
        vx_b = random.uniform(-2, 2)
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]  # Red = A
        img0[int(y), int(x_b)] = [0, 0, 1]  # Blue = B
        
        white = [1, 1, 1]
        
        # t1-t4: approach
        for _ in range(1, 5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            # Both bounce!
            if x_a < 3 or x_a > 29: vx_a *= -1
            if x_b < 3 or x_b > 29: vx_b *= -1
            x_a = max(3, min(28, x_a))
            x_b = max(3, min(28, x_b))
        
        # t5: full crossing - swap!
        x_a, x_b = x_b, x_a
        
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a)] = white
        img5[int(y), int(x_b)] = white
        
        # t6-t9: SAME physics + random reset
        for _ in range(6, 10):
            if random.random() < 0.5:
                vx_a = random.uniform(-2, 2)
            if random.random() < 0.5:
                vx_b = random.uniform(-2, 2)
            
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            
            # Both bounce!
            if x_a < 3 or x_a > 29: vx_a *= -1
            if x_b < 3 or x_b > 29: vx_b *= -1
            x_a = max(3, min(28, x_a))
            x_b = max(3, min(28, x_b))
        
        x_a = max(3, min(28, x_a))
        x_b = max(3, min(28, x_b))
        
        # t10: final
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x_a)] = white
        img10[int(y), int(x_b)] = white
        
        X_t0.append(img0)
        X_t5.append(img5)
        X_t10.append(img10)
        Y_a.append(x_a / 32)
    
    return np.array(X_t0), np.array(X_t5), np.array(X_t10), np.array(Y_a)

print("\n1. Generating data...")
X0, X5, X10, Y = generate_data(3000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   Data: {X0.shape}")

# Verify: same physics = harder to predict which is A
print("\n2. Verifying task difficulty...")

# Check if positions are predictable without identity
# If both balls have same physics, their final positions should be correlated
# with initial positions in complex ways
initial_pos_a = X0[:, 0, 16, :].argmax(axis=1).float()
initial_pos_b = X0[:, 2, 16, :].argmax(axis=1).float()
final_pos_a = Y.numpy()

# If identity doesn't matter, which ball is "A" is arbitrary
corr = np.corrcoef(initial_pos_a, final_pos_a)[0,1]
print(f"   Correlation (initial A -> final A): {corr:.3f}")
print(f"   (If correlation is random, identity is required!)")

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

print("\n3. Training Baseline...")
m = Baseline()
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
print(f"Baseline MSE: {mse:.4f} ({(random_mse-mse)/random_mse*100:.1f}% < random)")
print(f"Random MSE:   {random_mse:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse > random_mse * 0.7:
    print("=> Identity is REQUIRED!")
    print("=> Model cannot predict without tracking identity")
    print("=> This is the CORRECT benchmark!")
else:
    print("=> Some patterns still exploitable")
    print("=> Need further design")
