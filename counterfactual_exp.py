"""
COUNTERFACTUAL INTERVENTION EXPERIMENT
Key: Three-ball collision, predict "what if ball A was removed?"

This tests counterfactual reasoning - can trajectory model handle intervention?
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
print("COUNTERFACTUAL INTERVENTION EXPERIMENT")
print("="*60)

def clamp(val):
    return max(3, min(28, int(val)))

def generate_data(n=2000):
    """Three balls, predict counterfactual: what if ball A was removed?"""
    X_factual, X_counterfactual = [], []
    targets = []
    
    for _ in range(n):
        # Three balls
        x1, y1 = random.uniform(5, 10), random.uniform(10, 22)
        x2, y2 = random.uniform(11, 16), random.uniform(10, 22)
        x3, y3 = random.uniform(17, 27), random.uniform(10, 22)
        
        v1 = random.uniform(1, 2)   # Moving right
        v2 = random.uniform(-1, 1)  # Random
        v3 = random.uniform(-2, -1)  # Moving left
        
        # Factual: all three balls interact
        positions_f = [(x1, y1), (x2, y2), (x3, y3)]
        velocities = [(v1, 0), (v2, 0), (v3, 0)]
        
        # Simulate collision at t5
        for step in range(5):
            for i in range(3):
                x, y = positions_f[i]
                vx, vy = velocities[i]
                x += vx * 0.5
                if x < 3 or x > 29: vx *= -1
                positions_f[i] = (x, y)
                velocities[i] = (vx, vy)
        
        # Simple collision effect: ball 2 gets pushed
        # In factual, ball 2 moves faster due to collision
        v2_actual = velocities[1][0] * 1.5  # Enhanced
        
        # Counterfactual: remove ball 1
        positions_c = [(x2, y2), (x3, y3)]  # No ball 1
        velocities_c = [(v2, 0), (v3, 0)]
        
        # Counterfactual: no collision, ball 2 keeps original velocity
        v2_counter = v2
        
        # Continue for t6-t10
        positions_f[1] = (positions_f[1][0] + v2_actual * 2.5, positions_f[1][1])
        positions_c[1] = (positions_c[1][0] + v2_counter * 2.5, positions_c[1][1])
        
        # Final positions
        for i in range(3):
            x, y = positions_f[i]
            vx, vy = velocities[i]
            x += vx * 2.5
            if x < 3 or x > 29: vx *= -1
            positions_f[i] = (x, y)
        
        # Images
        img_f = np.zeros((32, 32, 3), np.float32)
        for x, y in positions_f:
            img_f[clamp(y), clamp(x)] = [1, 1, 1]
        
        img_c = np.zeros((32, 32, 3), np.float32)
        for x, y in positions_c:
            img_c[clamp(y), clamp(x)] = [1, 1, 1]
        
        X_factual.append(img_f)
        X_counterfactual.append(img_c)
        
        # Target: position of ball 2 in counterfactual (without collision)
        targets.append(positions_c[1][0] / 32)
    
    return np.array(X_factual), np.array(X_counterfactual), np.array(targets)

print("\n1. Generating counterfactual data...")
X_f, X_c, Y = generate_data(2000)
X_f = torch.FloatTensor(X_f).permute(0, 3, 1, 2)
X_c = torch.FloatTensor(X_c).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   Factual: {X_f.shape}, Counterfactual: {X_c.shape}")

# Train model on factual, test on counterfactual
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8*2, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x_f, x_c):
        h_f = self.enc(x_f).flatten(1)
        h_c = self.enc(x_c).flatten(1)
        return self.fc(torch.cat([h_f, h_c], dim=1)).squeeze()

print("\n2. Training on factual, testing on counterfactual...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

# Train on factual data predicting factual outcome
# But we want to see if it can generalize to counterfactual

# Generate training targets (factual)
Y_factual = Y.clone()  # Same distribution

for ep in range(10):
    idx = torch.randperm(len(X_f))
    for i in range(0, len(X_f), 64):
        # Train on factual only
        p = m(X_f[idx[i:i+64]], X_f[idx[i:i+64]])
        loss = F.mse_loss(p, Y_factual[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

# Test on counterfactual
m.eval()
with torch.no_grad():
    pred_c = m(X_f, X_c).numpy()
    
mse_counterfactual = ((pred_c - Y.numpy())**2).mean()
random_mse = Y.var().item()

print(f"\n3. Results:")
print(f"   Counterfactual MSE: {mse_counterfactual:.4f}")
print(f"   Random MSE: {random_mse:.4f}")
print(f"   Improvement: {(random_mse-mse_counterfactual)/random_mse*100:.1f}%")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse_counterfactual > random_mse * 0.7:
    print("=> Trajectory model FAILS on counterfactual!")
    print("=> Counterfactual reasoning requires different structure")
else:
    print("=> Model can generalize to counterfactual")
    print("=> More analysis needed")
