"""
IDENTITY-SWAP TEST

This is a decisive test to determine if model uses identity:
1. Train model on standard data (A=red→predict A)
2. Create swapped test set (swap A/B labels)
3. Test predictions

If model uses identity:
- Predictions will change significantly when labels swapped

If model doesn't use identity:
- Predictions will be nearly identical (model ignores identity)
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
print("IDENTITY-SWAP TEST")
print("="*60)

def generate_data(n=2000):
    """Generate identity tracking data"""
    X_t0, X_t5, X_t10 = [], [], []
    Y_a, Y_b = [], []
    
    for _ in range(n):
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        vx_a, vx_b = random.uniform(-2, 2), random.uniform(-2, 2)
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]
        img0[int(y), int(x_b)] = [0, 0, 1]
        
        white = [1, 1, 1]
        
        # Move to t5
        for _ in range(5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
        
        x_a = max(3, min(28, x_a))
        x_b = max(3, min(28, x_b))
        
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a)] = white
        img5[int(y), int(x_b)] = white
        
        # t10
        for _ in range(5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
        
        x_a = max(3, min(28, x_a))
        x_b = max(3, min(28, x_b))
        
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[int(y), int(x_a)] = white
        img10[int(y), int(x_b)] = white
        
        X_t0.append(img0)
        X_t5.append(img5)
        X_t10.append(img10)
        Y_a.append(x_a / 32)
        Y_b.append(x_b / 32)
    
    return np.array(X_t0), np.array(X_t5), np.array(X_t10), np.array(Y_a), np.array(Y_b)

print("\n1. Generating data...")
X0, X5, X10, Y_a, Y_b = generate_data(2000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
X10 = torch.FloatTensor(X10).permute(0, 3, 1, 2)
Y_a = torch.FloatTensor(Y_a)
Y_b = torch.FloatTensor(Y_b)

print(f"   Data: {X0.shape}")

# Model
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

print("\n2. Training model to predict A...")
m = Model()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]], X10[idx[i:i+64]])
        loss = F.mse_loss(p, Y_a[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()

# Test on original data (predict A)
pred_a = m(X0, X5, X10).detach().numpy()
mse_a = ((pred_a - Y_a.numpy())**2).mean()

# Test on SWAPPED data (swap A/B labels!)
# Now we ask: where is B?
# But model was trained to predict A's position
# If model uses identity, it should give wrong answer
pred_a_on_swapped = m(X0, X5, X10).detach().numpy()
# This is actually predicting B's position using A-trained model!

# Create swapped labels
Y_a_swapped = Y_b  # Now "A" label = old B position

mse_swapped = ((pred_a - Y_b.numpy())**2).mean()

# The key test: How different are the predictions?
corr = np.corrcoef(pred_a, Y_a.numpy())[0,1]
corr_swapped = np.corrcoef(pred_a, Y_b.numpy())[0,1]

print("\n" + "="*60)
print("IDENTITY-SWAP TEST RESULTS")
print("="*60)
print(f"Original (predict A → A): MSE = {mse_a:.4f}, Corr = {corr:.3f}")
print(f"Swapped  (predict A → B): MSE = {mse_swapped:.4f}, Corr = {corr_swapped:.3f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if corr > 0.8 and corr_swapped < 0.3:
    print("=> Model USES identity!")
    print("   - High correlation with A (correct)")
    print("   - Low correlation with B (wrong)")
elif corr > 0.5 and corr_swapped > 0.5:
    print("=> Model does NOT use identity!")
    print("   - Predictions similar for both A and B")
    print("   - Model learns average trajectory")
else:
    print("=> Results unclear, need analysis")
    print(f"   Corr(A): {corr:.3f}, Corr(B): {corr_swapped:.3f}")

# Additional test: What if we swap the images?
# Swap red/blue in t0 and see if predictions change
print("\n" + "="*60)
print("IMAGE SWAP TEST")
print("="*60)

# Swap colors in t0
X0_swapped = X0.clone()
# Swap red and blue channels
X0_swapped[:, 0], X0_swapped[:, 2] = X0[:, 2].clone(), X0[:, 0].clone()

pred_on_swapped_images = m(X0_swapped, X5, X10).detach().numpy()
mse_on_swapped_images = ((pred_on_swapped_images - Y_a.numpy())**2).mean()

print(f"Original images: MSE = {mse_a:.4f}")
print(f"Swapped colors:  MSE = {mse_on_swapped_images:.4f}")

if abs(mse_on_swapped_images - mse_a) < 0.001:
    print("=> Model IGNORES color identity!")
    print("   (MSE nearly identical regardless of color)")
else:
    print("=> Model USES color information")
