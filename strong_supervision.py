"""
L2 POSITIVE: Strong Supervision Approach

Core idea: Teach the model "what is the same object" through explicit supervision

Architecture:
1. Siamese encoder: shared weights for all frames
2. Identity binding loss: force same-object features to be similar across time
3. Explicit labels: know which object is which at training time
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
print("L2 POSITIVE: STRONG SUPERVISION")
print("="*60)

def generate_data(n=3000):
    """Generate identity tracking data with ground truth"""
    X_t0, X_t5 = [], []
    Y_a = []  # Position of ball A (was red at t0)
    
    for _ in range(n):
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        vx_a, vx_b = random.uniform(-2, 2), random.uniform(-2, 2)
        
        # t0: colored
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[int(y), int(x_a)] = [1, 0, 0]  # Red = A
        img0[int(y), int(x_b)] = [0, 0, 1]  # Blue = B
        
        white = [1, 1, 1]
        
        # Move to t5
        for _ in range(5):
            x_a += vx_a * 0.5
            x_b += vx_b * 0.5
            if x_a < 3 or x_a > 29: vx_a *= -1
        
        # t5: white balls - identity is ambiguous!
        img5 = np.zeros((32, 32, 3), np.float32)
        img5[int(y), int(x_a)] = white
        img5[int(y), int(x_b)] = white
        
        X_t0.append(img0)
        X_t5.append(img5)
        Y_a.append(x_a / 32)
    
    return np.array(X_t0), np.array(X_t5), np.array(Y_a)

print("\n1. Generating data...")
X0, X5, Y = generate_data(3000)
X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
X5 = torch.FloatTensor(X5).permute(0, 3, 1, 2)
Y = torch.FloatTensor(Y)

print(f"   X0: {X0.shape}, X5: {X5.shape}")

# ========== Model 1: Baseline ==========
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8*2, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x0, x5):
        return self.fc(torch.cat([self.enc(x0).flatten(1), self.enc(x5).flatten(1)], dim=1)).squeeze()

# ========== Model 2: Siamese + Identity Loss ==========
class SiameseIdentity(nn.Module):
    """
    Siamese network with identity consistency loss:
    - Shared encoder for all frames
    - Loss 1: Predict position of ball A (using t0 color info)
    - Loss 2: Identity consistency - enforce that "same object" features are similar
    """
    def __init__(self):
        super().__init__()
        # Shared encoder (Siamese)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
        )
        
        # Predictor
        self.predictor = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        
    def forward(self, x0, x5):
        # Encode both frames with SAME encoder
        h0 = self.encoder(x0)  # (B, 64)
        h5 = self.encoder(x5)  # (B, 64)
        
        # Predict position from t0 (has color info)
        pred = self.predictor(h0)
        
        return pred.squeeze(), h0, h5
    
    def identity_loss(self, h0, h5, is_same_object):
        """
        Force same object features to be similar
        is_same_object: Binary tensor (1 if same object, 0 if different)
        """
        # Cosine similarity
        cos = nn.CosineSimilarity(dim=1)
        similarity = cos(h0, h5)
        
        # If same object: maximize similarity
        # If different: minimize similarity
        loss = (is_same_object * (1 - similarity) + 
                (1 - is_same_object) * (1 + similarity))
        return loss.mean()

print("\n2. Training...")

# Baseline
print("   Training Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p = m(X0[idx[i:i+64]], X5[idx[i:i+64]])
        loss = F.mse_loss(p, Y[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_baseline = F.mse_loss(m(X0, X5), Y).item()

# Siamese with Identity Loss
print("   Training Siamese + Identity Loss...")
m = SiameseIdentity()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X0))
    for i in range(0, len(X0), 64):
        p, h0, h5 = m(X0[idx[i:i+64]], X5[idx[i:i+64]])
        
        # Main loss: predict correct position
        loss_main = F.mse_loss(p, Y[idx[i:i+64]])
        
        # Identity loss: all samples are "same object" (ball A at t0 == ball A at t5)
        is_same = torch.ones(len(p)).to(p.device)
        loss_identity = m.identity_loss(h0, h5, is_same)
        
        # Combined loss
        loss = loss_main + 0.1 * loss_identity
        opt.zero_grad(); loss.backward(); opt.step()

p, h0, h5 = m(X0, X5)
mse_siamese = F.mse_loss(p, Y).item()

random_mse = Y.var().item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Baseline:      MSE = {mse_baseline:.4f} ({(random_mse-mse_baseline)/random_mse*100:.1f}% < random)")
print(f"Siamese:       MSE = {mse_siamese:.4f} ({(random_mse-mse_siamese)/random_mse*100:.1f}% < random)")
print(f"Random:       MSE = {random_mse:.4f}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
# Check feature similarity
with torch.no_grad():
    cos = nn.CosineSimilarity(dim=1)
    sim_same = cos(h0, h5).mean().item()
    print(f"Feature similarity (same object): {sim_same:.4f}")
    print(f"(Higher = more similar = identity binding working)")
