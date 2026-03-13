"""
INTERACTION NETWORK BASELINE
Key: Object encoder + pairwise interaction + shared parameters
This enables combinatorial generalization through parameter sharing
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
print("INTERACTION NETWORK ON SCALING")
print("="*60)

def clamp(val):
    return max(3, min(28, int(val)))

def generate_data(n, n_objects):
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
        
        for step in range(10):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx * 0.5, y + vy * 0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i] = (x, y)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img10[clamp(y), clamp(x)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(positions[0][0] / 32)
    
    return np.array(X), np.array(Y)

print("\n1. Generating data...")
X_train, Y_train = generate_data(2000, 2)
X_test_2, Y_test_2 = generate_data(1000, 2)
X_test_6, Y_test_6 = generate_data(1000, 6)

X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
X_test_2 = torch.FloatTensor(X_test_2).permute(0, 3, 1, 2)
X_test_6 = torch.FloatTensor(X_test_6).permute(0, 3, 1, 2)
Y_train = torch.FloatTensor(Y_train)
Y_test_2 = torch.FloatTensor(Y_test_2)
Y_test_6 = torch.FloatTensor(Y_test_6)

print(f"   Train: {X_train.shape}, Test 2: {X_test_2.shape}, Test 6: {X_test_6.shape}")

# Model 1: Trajectory (Baseline)
class Trajectory(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 1))
    
    def forward(self, x):
        return self.fc(self.enc(x).flatten(1)).squeeze()

# Model 2: Interaction Network (object pairs)
class InteractionNet(nn.Module):
    """Object encoder + pairwise interaction + shared dynamics"""
    def __init__(self, max_objects=6):
        super().__init__()
        # Object encoder (shared across objects)
        self.obj_enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, 32),
        )
        
        # Pairwise interaction (shared parameters!)
        self.interact = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 32),
        )
        
        # Predictor
        self.predict = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    
    def forward(self, x):
        # x: (B, 6, 32, 32) - concatenate two frames
        B = x.size(0)
        
        # Extract objects (simplified: just encode both frames separately)
        f1 = x[:, :3]  # First frame
        f2 = x[:, 3:]   # Second frame
        
        # Encode each frame
        h1 = self.obj_enc(f1)  # (B, 32)
        h2 = self.obj_enc(f2)  # (B, 32)
        
        # Pairwise interaction
        h_pair = torch.cat([h1, h2], dim=1)  # (B, 64)
        h_inter = self.interact(h_pair)  # (B, 32)
        
        return self.predict(h_inter).squeeze()

print("\n2. Training Trajectory Model...")
m = Trajectory()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        p = m(X_train[idx[i:i+64]])
        loss = F.mse_loss(p, Y_train[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_traj_2 = F.mse_loss(m(X_test_2), Y_test_2).item()
mse_traj_6 = F.mse_loss(m(X_test_6), Y_test_6).item()

print("3. Training Interaction Net...")
m = InteractionNet()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        p = m(X_train[idx[i:i+64]])
        loss = F.mse_loss(p, Y_train[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_int_2 = F.mse_loss(m(X_test_2), Y_test_2).item()
mse_int_6 = F.mse_loss(m(X_test_6), Y_test_6).item()

random_2 = Y_test_2.var().item()
random_6 = Y_test_6.var().item()

print("\n" + "="*60)
print("RESULTS (MSE)")
print("="*60)
print(f"\nTrajectory Model:")
print(f"  2 objects: {mse_traj_2:.4f} ({(random_2-mse_traj_2)/random_2*100:.1f}%)")
print(f"  6 objects: {mse_traj_6:.4f} ({(random_6-mse_traj_6)/random_6*100:.1f}%)")
print(f"  Drop: {(mse_traj_6-mse_traj_2)/mse_traj_2*100:.1f}%")

print(f"\nInteraction Net:")
print(f"  2 objects: {mse_int_2:.4f} ({(random_2-mse_int_2)/random_2*100:.1f}%)")
print(f"  6 objects: {mse_int_6:.4f} ({(random_6-mse_int_6)/random_6*100:.1f}%)")
print(f"  Drop: {(mse_int_6-mse_int_2)/mse_int_2*100:.1f}%")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
traj_drop = (mse_traj_6-mse_traj_2)/mse_traj_2*100
int_drop = (mse_int_6-mse_int_2)/mse_int_2*100

if abs(int_drop) < abs(traj_drop):
    print(f"=> Interaction Net MORE stable! Drop: {int_drop:.1f}% vs Trajectory: {traj_drop:.1f}%")
else:
    print(f"=> Trajectory more stable. Need further analysis.")
