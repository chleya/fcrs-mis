"""
L4 EXPERIMENT: Intervention / Counterfactual Reasoning
Test: If we remove object A, what happens to object B?

Key question: Can object representations enable intervention simulation?
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
print("L4: INTERVENTION / COUNTERFACTUAL REASONING")
print("="*60)

def clamp(val):
    return max(3, min(28, int(val)))

def generate_intervention_data(n=2000):
    """
    Generate data with two scenarios:
    1. Factual: Two balls, ball A collides with ball B
    2. Counterfactual: Remove ball A, predict ball B's trajectory
    
    Training: Learn factual dynamics
    Testing: Predict counterfactual (what if A was removed?)
    """
    X_factual, Y_factual = [], []
    X_counter, Y_counter = [], []
    
    for _ in range(n):
        # Setup: Ball A moving right, Ball B stationary
        x_a, y_a = random.uniform(5, 12), random.uniform(10, 22)
        x_b, y_b = random.uniform(16, 22), random.uniform(10, 22)
        
        # Ensure same y-level for collision
        y_a = y_b = random.uniform(12, 20)
        
        v_a = random.uniform(2, 4)  # Moving right
        v_b = 0
        
        # === FACTUAL: Both balls ===
        # t0
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[clamp(y_a), clamp(x_a)] = [1, 0, 0]  # Red = A
        img0[clamp(y_b), clamp(x_b)] = [0, 0, 1]  # Blue = B
        
        # Simulate collision
        for step in range(10):
            x_a += v_a * 0.5
            x_b += v_b * 0.5
            
            # Simple collision: A hits B, B gets velocity
            if x_a >= x_b - 1 and v_a > 0:
                v_b = v_a * 0.8  # B gets momentum
                v_a = -v_a * 0.3  # A bounces back
                x_a = x_b - 1.1  # Separate
        
            # Bounce off walls
            if x_a < 3 or x_a > 29: v_a *= -1
            if x_b < 3 or x_b > 29: v_b *= -1
        
        # t10: final positions
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[clamp(y_a), clamp(x_a)] = [1, 1, 1]
        img10[clamp(y_b), clamp(x_b)] = [1, 1, 1]
        
        X_factual.append(np.concatenate([img0, img10], axis=2))
        
        # Target: final position of ball B
        Y_factual.append(x_b / 32)
        
        # === COUNTERFACTUAL: Remove ball A ===
        # Reset ball B to original position
        x_b_orig = x_b
        v_b_orig = 0
        
        # t10 without collision: B just moves straight
        for step in range(10):
            x_b_orig += v_b_orig * 0.5
            if x_b_orig < 3 or x_b_orig > 29: v_b_orig *= -1
        
        # Same image at t0 (ball A visible)
        img0_c = np.zeros((32, 32, 3), np.float32)
        img0_c[clamp(y_a), clamp(x_a)] = [1, 0, 0]
        img0_c[clamp(y_b), clamp(x_b)] = [0, 0, 1]
        
        # t10: what would happen WITHOUT A
        img10_c = np.zeros((32, 32, 3), np.float32)
        img10_c[clamp(y_b), clamp(x_b_orig)] = [1, 1, 1]
        
        X_counter.append(np.concatenate([img0_c, img10_c], axis=2))
        
        # Target: where B would be if A was never there
        Y_counter.append(x_b_orig / 32)
    
    return (np.array(X_factual, dtype=np.float32), np.array(Y_factual, dtype=np.float32),
            np.array(X_counter, dtype=np.float32), np.array(Y_counter, dtype=np.float32))

# Also generate regular prediction data for comparison
def generate_regular_data(n=2000, n_objects=2):
    X, Y = [], []
    for _ in range(n):
        positions = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for i, (x, y) in enumerate(positions):
            if i == 0:
                img0[clamp(y), clamp(x)] = [1, 0, 0]
            else:
                img0[clamp(y), clamp(x)] = [0, 0, 1]
        
        for _ in range(10):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i], velocities[i] = (x, y), (vx, vy)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img10[clamp(y), clamp(x)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(positions[0][0] / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Coordinates version
def make_features(positions, velocities):
    feat = []
    for i in range(MAX_OBJ):
        if i < len(positions):
            x, y = positions[i]
            vx, vy = velocities[i]
            feat.extend([x/32, y/32, vx/4, vy/4])
        else:
            feat.extend([0, 0, 0, 0])
    return feat

def generate_coord_data(n=2000, n_objects=2):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        f0 = make_features(pos, vel)
        
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]
                vx, vy = vel[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                pos[i], vel[i] = (x, y), (vx, vy)
        
        f10 = make_features(pos, vel)
        X.append(f0 + f10)
        Y.append(pos[0][0] / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("\n1. Generating data...")

# Regular training data
X_reg_tr, Y_reg_tr = generate_regular_data(2000, 2)
X_reg_t, Y_reg_t = generate_regular_data(1000, 2)

# Intervention data
Xf_tr, Yf_tr, Xc_tr, Yc_tr = generate_intervention_data(2000)
Xf_t, Yf_t, Xc_t, Yc_t = generate_intervention_data(1000)

# Coordinates
X_coord_tr, Y_coord_tr = generate_coord_data(2000, 2)
X_coord_t, Y_coord_t = generate_coord_data(1000, 2)

# Convert to tensors
X_reg_tr = torch.FloatTensor(X_reg_tr).permute(0, 3, 1, 2)
X_reg_t = torch.FloatTensor(X_reg_t).permute(0, 3, 1, 2)
Xf_tr = torch.FloatTensor(Xf_tr).permute(0, 3, 1, 2)
Xf_t = torch.FloatTensor(Xf_t).permute(0, 3, 1, 2)
Xc_tr = torch.FloatTensor(Xc_tr).permute(0, 3, 1, 2)
Xc_t = torch.FloatTensor(Xc_t).permute(0, 3, 1, 2)

X_coord_tr = torch.FloatTensor(X_coord_tr)
X_coord_t = torch.FloatTensor(X_coord_t)

Y_reg_tr = torch.FloatTensor(Y_reg_tr)
Yf_tr = torch.FloatTensor(Yf_tr)
Yc_tr = torch.FloatTensor(Yc_tr)
Y_coord_tr = torch.FloatTensor(Y_coord_tr)

print(f"   Regular: {X_reg_tr.shape}")
print(f"   Factual: {Xf_tr.shape}, Counterfactual: {Xc_tr.shape}")
print(f"   Coordinates: {X_coord_tr.shape}")

# Models
class TrajPix(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(self.enc(x).flatten(1)).squeeze()

class ObjPix(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.slots = nn.Parameter(torch.randn(4, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        h = self.enc(x).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

class TrajCoord(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(x).squeeze()

class ObjCoord(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16))
        self.out = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 2, MAX_OBJ, 4)
        h = self.proc(x[:, 1])
        return self.out(h[:, 0]).squeeze()

def train_and_eval(model, X_tr, Y_tr, X_te, Y_te):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            p = model(X_tr[idx[i:i+64]])
            loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    mse = F.mse_loss(model(X_te), Y_te).item()
    rnd = Y_te.var().item()
    return mse, rnd

print("\n2. Running experiments...")

# Experiment 1: Regular prediction (baseline)
print("\n--- Regular Prediction (baseline) ---")
mse, rnd = train_and_eval(TrajPix(), X_reg_tr, Y_reg_tr, X_reg_t, Y_reg_t)
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = train_and_eval(ObjPix(), X_reg_tr, Y_reg_tr, X_reg_t, Y_reg_t)
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

# Experiment 2: Counterfactual prediction
print("\n--- Counterfactual: Train factual → Test counterfactual ---")

# Trajectory on pixel
mse, rnd = train_and_eval(TrajPix(), Xf_tr, Yf_tr, Xc_t, Yc_t)
print(f"Traj(Pixel): {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

# Object on pixel
mse, rnd = train_and_eval(ObjPix(), Xf_tr, Yf_tr, Xc_t, Yc_t)
print(f"Obj(Pixel):  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

# Coordinates
mse, rnd = train_and_eval(TrajCoord(), X_coord_tr, Y_coord_tr, X_coord_t, Y_coord_t)
print(f"Traj(Coord): {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

mse, rnd = train_and_eval(ObjCoord(), X_coord_tr, Y_coord_tr, X_coord_t, Y_coord_t)
print(f"Obj(Coord):  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("\nCounterfactual reasoning requires understanding:")
print("  1. Objects are separate entities")
print("  2. Dynamics depend on objects")
print("  3. Removing object changes dynamics")
print("\nIf models can do this, they understand object-level causation.")
