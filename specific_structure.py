"""
L2 POSITIVE: More Specific Structure
Instead of weak slot+bias, use stronger structure:
1. Two separate feature extractors (one per object)
2. Explicit color-based routing
3. Direct supervision with identity labels
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
print("L2 POSITIVE: SPECIFIC STRUCTURE TEST")
print("="*60)

def generate_data(n=3000):
    """Simple: t0 colored, t5 white, predict A"""
    X_t0, X_t5 = [], []
    Y_a, Y_b = [], []
    
    for _ in range(n):
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        vx_a, vx_b = random.uniform(-2, 2), random.uniform(-2, 2)
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]  # Red
        img0[int(y), int(x_b)] = [0, 0, 1]  # Blue
        
        white = [1, 1, 1]
        
        # Move to t5
        for _ in range(5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
        
        # t5: white
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a)] = white
        img5[int(y), int(x_b)] = white
        
        X_t0.append(img0)
        X_t5.append(img5)
        Y_a.append(x_a / 32)
        Y_b.append(x_b / 32)
    
    return np.array(X_t0), np.array(X_t5), np.array(Y_a), np.array(Y_b)

print("\n1. Generating data...")
X0, X5, Y_a, Y_b = generate_data(3000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
Y_a = torch.FloatTensor(Y_a)
Y_b = torch.FloatTensor(Y_b)

print(f"   Data: {X0.shape}")

# ========== Model 1: Baseline ==========
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8*2, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x0, x5):
        return self.fc(torch.cat([self.enc(x0).flatten(1), self.enc(x5).flatten(1)], dim=1)).squeeze()

# ========== Model 2: Explicit Color Routing ==========
class ColorRouter(nn.Module):
    """
    Explicit structure:
    - Red channel → encodes ball A
    - Blue channel → encodes ball B
    - Predicts both positions, we select A
    """
    def __init__(self):
        super().__init__()
        # Separate encoders for R and B channels
        self.enc_red = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        )
        self.enc_blue = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        )
        # Shared encoder for white balls
        self.enc_white = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        # Predict positions
        self.predict = nn.Sequential(nn.Linear(64*8*8 + 32*8*8*2, 128), nn.ReLU(), nn.Linear(128, 2))
        
    def forward(self, x0, x5):
        # Extract red channel (ball A)
        red = x0[:, 0:1, :, :]  # (B, 1, 32, 32)
        h_red = self.enc_red(red).flatten(1)
        
        # Extract blue channel (ball B)
        blue = x0[:, 2:3, :, :]  # (B, 1, 32, 32)
        h_blue = self.enc_blue(blue).flatten(1)
        
        # White ball features
        h_white = self.enc_white(x5).flatten(1)
        
        # Combine and predict both positions
        h = torch.cat([h_white, h_red, h_blue], dim=1)
        out = self.predict(h)
        
        return out[:, 0]  # Return position A

# ========== Model 3: Direct Supervision ==========
class DirectSupervision(nn.Module):
    """
    Train with explicit supervision:
    - Input: t0 + t5
    - Output: position of ball that was RED at t0
    - Uses separate pathways for color vs white
    """
    def __init__(self):
        super().__init__()
        # Color encoder (detects red ball position)
        self.enc_color = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
        )
        # White encoder
        self.enc_white = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
        )
        # Combined predictor
        self.predict = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        
    def forward(self, x0, x5):
        h_color = self.enc_color(x0)
        h_white = self.enc_white(x5)
        h = torch.cat([h_color, h_white], dim=1)
        return self.predict(h).squeeze()

print("\n2. Training...")

# Baseline
print("   Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]])
        loss = F.mse_loss(p, Y_a[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_baseline = F.mse_loss(m(X0, X5), Y_a).item()

# Color Router
print("   ColorRouter...")
m = ColorRouter()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]])
        loss = F.mse_loss(p, Y_a[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_router = F.mse_loss(m(X0, X5), Y_a).item()

# Direct Supervision
print("   DirectSupervision...")
m = DirectSupervision()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]])
        loss = F.mse_loss(p, Y_a[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_direct = F.mse_loss(m(X0, X5), Y_a).item()

random_mse = Y_a.var().item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Baseline:          MSE = {mse_baseline:.4f} ({(random_mse-mse_baseline)/random_mse*100:.1f}% < random)")
print(f"ColorRouter:      MSE = {mse_router:.4f} ({(random_mse-mse_router)/random_mse*100:.1f}% < random)")
print(f"DirectSupervision: MSE = {mse_direct:.4f} ({(random_mse-mse_direct)/random_mse*100:.1f}% < random)")
print(f"Random:           MSE = {random_mse:.4f}")
