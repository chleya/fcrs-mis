"""
STAGE 2: Visual Cues Experiment
Test: How do visual cues affect object representation emergence?

Conditions:
1. same color - all white balls (current setting)
2. distinct colors - each object different color

Hypothesis: Distinct colors should make object learning easier
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
print("STAGE 2: VISUAL CUES EXPERIMENT")
print("="*60)

def clamp(val):
    return max(3, min(28, int(val)))

def generate_data(n, n_objects, color_mode='same'):
    """
    color_mode: 'same' (all white) or 'distinct' (each object different color)
    """
    X, Y = [], []
    
    # Color palette for distinct mode
    color_palette = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
    ]
    
    for _ in range(n):
        positions = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_objects)]
        velocities = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_objects)]
        
        # t0: colored balls
        img0 = np.zeros((32, 32, 3), np.float32)
        for i, (x, y) in enumerate(positions):
            if color_mode == 'distinct' and i < len(color_palette):
                img0[clamp(y), clamp(x)] = color_palette[i]
            else:
                img0[clamp(y), clamp(x)] = [1, 1, 1]  # White
        
        # Move
        for step in range(10):
            for i in range(len(positions)):
                x, y = positions[i]
                vx, vy = velocities[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                positions[i], velocities[i] = (x, y), (vx, vy)
        
        # t10: all white
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in positions:
            img10[clamp(y), clamp(x)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(positions[0][0] / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Generate data
print("\n1. Generating data...")
X_train_same, Y_train_same = generate_data(2000, 2, 'same')
X_train_dist, Y_train_dist = generate_data(2000, 2, 'distinct')
X_test_same, Y_test_same = generate_data(1000, 6, 'same')
X_test_dist, Y_test_dist = generate_data(1000, 6, 'distinct')

X_train_same = torch.FloatTensor(X_train_same).permute(0, 3, 1, 2)
X_train_dist = torch.FloatTensor(X_train_dist).permute(0, 3, 1, 2)
X_test_same = torch.FloatTensor(X_test_same).permute(0, 3, 1, 2)
X_test_dist = torch.FloatTensor(X_test_dist).permute(0, 3, 1, 2)
Y_train_same = torch.FloatTensor(Y_train_same)
Y_train_dist = torch.FloatTensor(Y_train_dist)
Y_test_same = torch.FloatTensor(Y_test_same)
Y_test_dist = torch.FloatTensor(Y_test_dist)

print(f"   Same color: Train {X_train_same.shape}, Test {X_test_same.shape}")
print(f"   Distinct:   Train {X_train_dist.shape}, Test {X_test_dist.shape}")

# Models
class TrajectoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 1))
    
    def forward(self, x):
        return self.fc(self.enc(x).flatten(1)).squeeze()

class SlotModel(nn.Module):
    def __init__(self, n_slots=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slots = nn.Parameter(torch.randn(n_slots, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x):
        h = self.enc(x).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

print("\n2. Training and testing...")

# Function to train and evaluate
def run_experiment(X_train, Y_train, X_test, Y_test, name):
    # Train Trajectory
    m = TrajectoryModel()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 64):
            p = m(X_train[idx[i:i+64]])
            loss = F.mse_loss(p, Y_train[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    mse_traj = F.mse_loss(m(X_test), Y_test).item()
    
    # Train Slot
    m = SlotModel(n_slots=4)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 64):
            p = m(X_train[idx[i:i+64]])
            loss = F.mse_loss(p, Y_train[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    
    mse_slot = F.mse_loss(m(X_test), Y_test).item()
    
    random_mse = Y_test.var().item()
    
    return mse_traj, mse_slot, random_mse

# Run experiments
print("\n--- Same Color (all white) ---")
mse_t_same, mse_s_same, rnd_same = run_experiment(
    X_train_same, Y_train_same, X_test_same, Y_test_same, "same")

print(f"Trajectory: MSE={mse_t_same:.4f} ({(rnd_same-mse_t_same)/rnd_same*100:.0f}%)")
print(f"Slot:      MSE={mse_s_same:.4f} ({(rnd_same-mse_s_same)/rnd_same*100:.0f}%)")
print(f"Random:    MSE={rnd_same:.4f}")

print("\n--- Distinct Colors ---")
mse_t_dist, mse_s_dist, rnd_dist = run_experiment(
    X_train_dist, Y_train_dist, X_test_dist, Y_test_dist, "distinct")

print(f"Trajectory: MSE={mse_t_dist:.4f} ({(rnd_dist-mse_t_dist)/rnd_dist*100:.0f}%)")
print(f"Slot:      MSE={mse_s_dist:.4f} ({(rnd_dist-mse_s_dist)/rnd_dist*100:.0f}%)")
print(f"Random:    MSE={rnd_dist:.4f}")

print("\n" + "="*60)
print("COMPARISON: Visual Cues Effect")
print("="*60)
print(f"\nSame Color → 6 objects:")
print(f"  Trajectory: {(rnd_same-mse_t_same)/rnd_same*100:.0f}%")
print(f"  Slot:       {(rnd_same-mse_s_same)/rnd_same*100:.0f}%")

print(f"\nDistinct Colors → 6 objects:")
print(f"  Trajectory: {(rnd_dist-mse_t_dist)/rnd_dist*100:.0f}%")
print(f"  Slot:       {(rnd_dist-mse_s_dist)/rnd_dist*100:.0f}%")

slot_diff = ((rnd_dist-mse_s_dist)/rnd_dist*100) - ((rnd_same-mse_s_same)/rnd_same*100)
print(f"\nSlot improvement with distinct colors: {slot_diff:.0f}%")

if slot_diff > 20:
    print("\n=> Distinct colors HELP Slot learn! Visual cues matter!")
else:
    print("\n=> Visual cues have limited effect - need stronger signals")
