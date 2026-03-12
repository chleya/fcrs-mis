"""
INFORMATION THEORY EXPERIMENT
Measure: I(object identity; prediction)

If I ≈ 0: object structure is redundant
If I > 0: object structure is necessary
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
print("INFORMATION THEORY EXPERIMENT")
print("="*60)

def generate_data(n=3000):
    """Generate data with object identity"""
    X_t0, X_t10 = [], []
    Y_a, Y_b = [], []
    
    for _ in range(n):
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        vx_a, vx_b = random.uniform(-2, 2), random.uniform(-2, 2)
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]  # Red = A
        img0[int(y), int(x_b)] = [0, 0, 1]  # Blue = B
        
        white = [1, 1, 1]
        
        # Move to t10
        for _ in range(10):
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
        X_t10.append(img10)
        Y_a.append(x_a / 32)
        Y_b.append(x_b / 32)
    
    return np.array(X_t0), np.array(X_t10), np.array(Y_a), np.array(Y_b)

print("\n1. Generating data...")
X0, X10, Y_a, Y_b = generate_data(3000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y_a = torch.FloatTensor(Y_a)
Y_b = torch.FloatTensor(Y_b)

print(f"   Data: {X0.shape}")

# Train model to predict A
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
    
    def forward(self, x0, x10):
        h0 = self.enc(x0).flatten(1)
        h10 = self.enc(x10).flatten(1)
        return self.fc(torch.cat([h0, h10], dim=1)).squeeze()

print("\n2. Training model...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y_a[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()

print("\n3. Computing information measures...")

# Get predictions
with torch.no_grad():
    pred_a = m(X0, X10).numpy()
    pred_b = m(X0.flip(1), X10).numpy()  # Swap colors

# Compute mutual information approximation
# I(X; Y) ≈ H(Y) - H(Y|X)

# H(Y) - entropy of targets
def entropy(x, bins=20):
    """Compute entropy"""
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist + 1e-10
    return -np.sum(hist * np.log(hist))

H_Y_a = entropy(Y_a.numpy())
H_Y_b = entropy(Y_b.numpy())

# H(Y|X) - conditional entropy (residual variance)
residuals_a = pred_a - Y_a.numpy()
residuals_b = pred_b - Y_b.numpy()

H_Y_givenX_a = entropy(residuals_a)
H_Y_givenX_b = entropy(residuals_b)

# Mutual information
I_a = H_Y_a - H_Y_givenX_a
I_b = H_Y_b - H_Y_givenX_b

print(f"\n   H(Y_a) = {H_Y_a:.4f}")
print(f"   H(Y_a|X) = {H_Y_givenX_a:.4f}")
print(f"   I(object_a; prediction) = {I_a:.4f}")

print(f"\n   H(Y_b) = {H_Y_b:.4f}")
print(f"   H(Y_b|X) = {H_Y_givenX_b:.4f}")
print(f"   I(object_b; prediction) = {I_b:.4f}")

# Normalized mutual information (0-1)
NMI_a = I_a / (H_Y_a + 1e-10)
NMI_b = I_b / (H_Y_b + 1e-10)

print(f"\n   NMI (normalized): {NMI_a:.4f}, {NMI_b:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print(f"\nIf I ≈ 0: Object identity is NOT used by model")
print(f"If I > 0: Object identity IS used by model")

if I_a > 0.5:
    print(f"\n=> Model uses object identity! I = {I_a:.4f}")
else:
    print(f"\n=> Model does NOT use object identity! I = {I_a:.4f}")

# Compare: how much does swapping affect predictions?
diff = np.abs(pred_a - pred_b).mean()
print(f"\nPrediction difference when swapping colors: {diff:.4f}")
print("(Higher = model uses color/object identity)")
