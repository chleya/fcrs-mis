"""
L5: Object + Causal Training
Train on: prediction + intervention tasks
Test: Does causal training enable intervention reasoning?

Hypothesis: Object representation + causal training → intervention reasoning
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MAX_OBJ = 6
print("="*60)
print("L5: CAUSAL TRAINING EXPERIMENT")
print("="*60)

def clamp(v): return max(3, min(28, int(v)))

# Standard prediction data
def gen_pred(n, n_obj=2):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_obj)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_obj)]
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for i, (x, y) in enumerate(pos):
            img0[clamp(y), clamp(x)] = [1, 1, 1]
        
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]
                vx, vy = vel[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                pos[i], vel[i] = (x, y), (vx, vy)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos:
            img10[clamp(y), clamp(x)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(pos[0][0] / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Intervention data: remove one object
def gen_intervention(n):
    """
    Two scenarios:
    1. Normal: both objects present
    2. Intervention: object removed
    
    Target: predict where object B ends up
    """
    X_norm, Y_norm = [], []
    X_int, Y_int = [], []
    
    for _ in range(n):
        # Two balls: A (moving), B (stationary)
        xa, ya = random.uniform(5, 12), random.uniform(12, 20)
        xb, yb = random.uniform(16, 22), ya
        va = random.uniform(2, 4)
        vb = 0
        
        # === Normal scenario ===
        xb_n = xb
        for _ in range(10):
            xa += va * 0.5
            xb_n += vb * 0.5
            if xa >= xb_n - 1 and va > 0:
                vb = va * 0.8
                va = -va * 0.3
                xa = xb_n - 1.1
        
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[clamp(ya), clamp(xa)] = [1, 0, 0]
        img0[clamp(yb), clamp(xb)] = [0, 0, 1]
        
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[clamp(ya), clamp(xa)] = [1, 1, 1]
        img10[clamp(yb), clamp(xb_n)] = [1, 1, 1]
        
        X_norm.append(np.concatenate([img0, img10], axis=2))
        Y_norm.append(xb_n / 32)
        
        # === Intervention: remove A ===
        xb_i = xb
        for _ in range(10):
            xb_i += 0 * 0.5  # No collision
        
        # Input still shows both (but A will be "removed" in concept)
        # Actually: input shows only B at t0
        img0_i = np.zeros((32, 32, 3), np.float32)
        img0_i[clamp(yb), clamp(xb)] = [0, 0, 1]  # Only B visible
        
        img10_i = np.zeros((32, 32, 3), np.float32)
        img10_i[clamp(yb), clamp(xb_i)] = [1, 1, 1]
        
        X_int.append(np.concatenate([img0_i, img10_i], axis=2))
        Y_int.append(xb_i / 32)
    
    return (np.array(X_norm, dtype=np.float32), np.array(Y_norm, dtype=np.float32),
            np.array(X_int, dtype=np.float32), np.array(Y_int, dtype=np.float32))

print("\n1. Generating data...")

# Training: prediction + intervention MIX
X_pred_tr, Y_pred_tr = gen_pred(2000, 2)
Xn_tr, Yn_tr, Xi_tr, Yi_tr = gen_intervention(2000)

# Test: intervention only
Xn_t, Yn_t, Xi_t, Yi_t = gen_intervention(1000)

# Convert
X_pred_tr = torch.FloatTensor(X_pred_tr).permute(0, 3, 1, 2)
Xn_tr = torch.FloatTensor(Xn_tr).permute(0, 3, 1, 2)
Xi_tr = torch.FloatTensor(Xi_tr).permute(0, 3, 1, 2)

Xn_t = torch.FloatTensor(Xn_t).permute(0, 3, 1, 2)
Xi_t = torch.FloatTensor(Xi_t).permute(0, 3, 1, 2)

Y_pred_tr = torch.FloatTensor(Y_pred_tr)
Yn_tr = torch.FloatTensor(Yn_tr)
Yi_tr = torch.FloatTensor(Yi_tr)

print(f"   Prediction: {X_pred_tr.shape}")
print(f"   Normal: {Xn_tr.shape}, Intervention: {Xi_tr.shape}")

# Models
class TrajModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(self.enc(x).flatten(1)).squeeze()

class ObjModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.slots = nn.Parameter(torch.randn(4, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        h = self.enc(x).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

def train_pred_only(model, X, Y, X_test, Y_test, name):
    """Train on prediction only"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            p = model(X[idx[i:i+64]])
            loss = F.mse_loss(p, Y[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    mse = ((model(X_test) - Y_test) ** 2).mean().item()
    rnd = Y_test.var().item()
    return mse, rnd

def train_mixed(model, X_pred, Y_pred, X_int, Y_int, X_test, Y_test, name):
    """Train on prediction + intervention (mixed)"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Mix: 50% prediction, 50% intervention
    for ep in range(10):
        # Prediction batch
        idx_p = torch.randperm(len(X_pred))
        # Intervention batch
        idx_i = torch.randperm(len(X_int))
        
        # Train on both
        for i in range(0, min(len(X_pred), len(X_int)), 32):
            # Prediction
            p_pred = model(X_pred[idx_p[i:i+32]])
            loss_pred = F.mse_loss(p_pred, Y_pred[idx_p[i:i+32]])
            
            # Intervention
            p_int = model(X_int[idx_i[i:i+32]])
            loss_int = F.mse_loss(p_int, Y_int[idx_i[i:i+32]])
            
            loss = loss_pred + loss_int
            opt.zero_grad(); loss.backward(); opt.step()
    
    mse = ((model(X_test) - Y_test) ** 2).mean().item()
    rnd = Y_test.var().item()
    return mse, rnd

print("\n2. Running experiments...")

# Baseline: prediction only → intervention test
print("\n--- Baseline: Prediction only → Intervention test ---")
mse, rnd = train_pred_only(TrajModel(), X_pred_tr, Y_pred_tr, Xi_t, Yi_t, "Traj")
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = train_pred_only(ObjModel(), X_pred_tr, Y_pred_tr, Xi_t, Yi_t, "Obj")
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

# Causal training: prediction + intervention
print("\n--- Causal Training: Prediction + Intervention → Intervention test ---")
mse, rnd = train_mixed(TrajModel(), X_pred_tr, Y_pred_tr, Xi_tr, Yi_tr, Xi_t, Yi_t, "Traj")
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = train_mixed(ObjModel(), X_pred_tr, Y_pred_tr, Xi_tr, Yi_tr, Xi_t, Yi_t, "Obj")
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("\nIf causal training helps:")
print("  → Intervention accuracy improves")
print("  → Object model with causal training can reason about removal")
print("\nIf not:")
print("  → Both models still fail")
print("  → Need stronger causal structure (e.g., physics engine)")
