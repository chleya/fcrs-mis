"""
CONTROL EXPERIMENT: Coordinates Input
Remove perception - use ground-truth (x, y, vx, vy)
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
print("COORDINATES INPUT: REMOVE PERCEPTION")
print("="*60)

MAX_OBJECTS = 6

def generate_data_padded(n, n_objects):
    """Generate ground-truth coordinates, pad to MAX_OBJECTS"""
    X, Y = [], []
    for _ in range(n):
        positions = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        # t0 features - pad to MAX_OBJECTS
        features_0 = []
        for i in range(n_objects):
            x, y = positions[i]
            vx, vy = velocities[i]
            features_0.extend([x/32, y/32, vx/4, vy/4])
        # Pad with zeros
        while len(features_0) < MAX_OBJECTS * 4:
            features_0.extend([0, 0, 0, 0])
        
        # Move
        for step in range(10):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx * 0.5, y + vy * 0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i] = (x, y)
                velocities[i] = (vx, vy)
        
        # t10 features - pad to MAX_OBJECTS
        features_10 = []
        for i in range(n_objects):
            x, y = positions[i]
            vx, vy = velocities[i]
            features_10.extend([x/32, y/32, vx/4, vy/4])
        while len(features_10) < MAX_OBJECTS * 4:
            features_10.extend([0, 0, 0, 0])
        
        X.append(features_0 + features_10)
        Y.append(positions[0][0] / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("\n1. Generating data...")
X_train, Y_train = generate_data_padded(2000, 2)  # Pad 2 objects to 6
X_test_2, Y_test_2 = generate_data_padded(1000, 2)
X_test_6, Y_test_6 = generate_data_padded(1000, 6)

X_train = torch.FloatTensor(X_train)
X_test_2 = torch.FloatTensor(X_test_2)
X_test_6 = torch.FloatTensor(X_test_6)
Y_train = torch.FloatTensor(Y_train)
Y_test_2 = torch.FloatTensor(Y_test_2)
Y_test_6 = torch.FloatTensor(Y_test_6)

print(f"   Train: {X_train.shape}")
print(f"   Test 2: {X_test_2.shape}")
print(f"   Test 6: {X_test_6.shape}")

# Model 1: Trajectory
class TrajectoryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(48, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(x).squeeze()

# Model 2: Object-Factored (shared params)
class ObjectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_proc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16))
        self.agg = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x):
        B = x.size(0)
        objects = x.view(B, MAX_OBJECTS, 4)
        h = self.obj_proc(objects)
        return self.agg(h[:, 0]).squeeze()

print("\n2. Training Trajectory MLP...")
m = TrajectoryMLP()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        p = m(X_train[idx[i:i+64]])
        loss = F.mse_loss(p, Y_train[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_traj_2 = F.mse_loss(m(X_test_2), Y_test_2).item()
mse_traj_6 = F.mse_loss(m(X_test_6), Y_test_6).item()

print("3. Training Object MLP...")
m = ObjectMLP()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        p = m(X_train[idx[i:i+64]])
        loss = F.mse_loss(p, Y_train[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

mse_obj_2 = F.mse_loss(m(X_test_2), Y_test_2).item()
mse_obj_6 = F.mse_loss(m(X_test_6), Y_test_6).item()

random_2 = Y_test_2.var().item()
random_6 = Y_test_6.var().item()

print("\n" + "="*60)
print("RESULTS (Coordinates Input - MSE)")
print("="*60)
print(f"\nTrajectory MLP:")
print(f"  2 objects: {mse_traj_2:.4f} ({(random_2-mse_traj_2)/random_2*100:.1f}%)")
print(f"  6 objects: {mse_traj_6:.4f} ({(random_6-mse_traj_6)/random_6*100:.1f}%)")
print(f"  Drop: {(mse_traj_6-mse_traj_2)/mse_traj_2*100:.1f}%")

print(f"\nObject MLP:")
print(f"  2 objects: {mse_obj_2:.4f} ({(random_2-mse_obj_2)/random_2*100:.1f}%)")
print(f"  6 objects: {mse_obj_6:.4f} ({(random_6-mse_obj_6)/random_6*100:.1f}%)")
print(f"  Drop: {(mse_obj_6-mse_obj_2)/mse_obj_2*100:.1f}%")

traj_drop = (mse_traj_6-mse_traj_2)/mse_traj_2*100
obj_drop = (mse_obj_6-mse_obj_2)/mse_obj_2*100

print(f"\nTrajectory Drop: {traj_drop:.1f}%")
print(f"Object Drop: {obj_drop:.1f}%")

if obj_drop < traj_drop - 30:
    print("\n=> Object MORE stable!")
else:
    print("\n=> Both fail similarly")
