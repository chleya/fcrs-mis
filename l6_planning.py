"""
L6: PLANNING EXPERIMENT
Test: Can object representation enable planning?

Task: Push ball A to reach target position
This requires: world model + trajectory optimization

Question: Does object representation help with planning?
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
print("L6: PLANNING EXPERIMENT")
print("="*60)

def clamp(v): return max(3, min(28, int(v)))

# Planning data: reach target
def gen_planning(n):
    """
    Task: Given ball position + target, find action to reach target
    
    Input: Ball start position, target position
    Output: Velocity command
    
    This tests if model can plan trajectories
    """
    X, Y = [], []
    for _ in range(n):
        # Ball start position
        bx = random.uniform(5, 20)
        by = random.uniform(8, 24)
        
        # Target position
        tx = random.uniform(10, 27)
        ty = random.uniform(8, 24)
        
        # Simple planning: velocity toward target
        dx = tx - bx
        dy = ty - by
        vx = dx / 10  # Normalize to reach in 10 steps
        vy = dy / 10
        
        # Input: two frames showing movement
        img0 = np.zeros((32, 32, 3), np.float32)
        img0[clamp(by), clamp(bx)] = [1, 0, 0]  # Ball = red
        # Mark target with different color in corner
        img0[2, 2] = [0, 1, 0]  # Target = green marker
        
        # Move ball toward target
        for _ in range(10):
            bx += vx
            by += vy
        
        img10 = np.zeros((32, 32, 3), np.float32)
        img10[clamp(by), clamp(bx)] = [1, 1, 1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        
        # Target: final position
        Y.append(bx / 32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Standard prediction (no planning)
def gen_pred(n):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(2)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(2)]
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img0[clamp(y), clamp(x)] = [1,1,1]
        
        for _ in range(10):
            for i in range(2):
                x, y = pos[i]; vx, vy = vel[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img10[clamp(y), clamp(x)] = [1,1,1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(pos[0][0]/32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

print("\n1. Generating data...")

# Planning task
X_plan, Y_plan = gen_planning(2000)
X_plan_t, Y_plan_t = gen_planning(1000)

# Standard prediction
X_pred, Y_pred = gen_pred(2000)
X_pred_t, Y_pred_t = gen_pred(1000)

X_plan = torch.FloatTensor(X_plan).permute(0,3,1,2)
X_plan_t = torch.FloatTensor(X_plan_t).permute(0,3,1,2)
X_pred = torch.FloatTensor(X_pred).permute(0,3,1,2)
X_pred_t = torch.FloatTensor(X_pred_t).permute(0,3,1,2)

Y_plan = torch.FloatTensor(Y_plan)
Y_plan_t = torch.FloatTensor(Y_plan_t)
Y_pred = torch.FloatTensor(Y_pred)
Y_pred_t = torch.FloatTensor(Y_pred_t)

print(f"   Planning: {X_plan.shape}")
print(f"   Prediction: {X_pred.shape}")

# Models
class TrajModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x): return self.fc(self.enc(x).flatten(1)).squeeze()

class ObjModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(6,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU())
        self.slots = nn.Parameter(torch.randn(4,64)*0.1)
        self.predict = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x):
        h = self.enc(x).mean(dim=[2,3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:,0]).squeeze()

def run(model, Xtr, Ytr, Xte, Yte, name):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(10):
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            p = model(Xtr[idx[i:i+64]])
            loss = F.mse_loss(p, Ytr[idx[i:i+64]])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        mse = F.mse_loss(model(Xte), Yte).item()
    rnd = Yte.var().item()
    return mse, rnd

print("\n2. Running experiments...")

# Baseline: prediction → planning
print("\n--- Baseline: Prediction → Planning ---")
mse, rnd = run(TrajModel(), X_pred, Y_pred, X_plan_t, Y_plan_t, "Traj")
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = run(ObjModel(), X_pred, Y_pred, X_plan_t, Y_plan_t, "Obj")
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

# Direct planning training
print("\n--- Direct Planning Training ---")
mse, rnd = run(TrajModel(), X_plan, Y_plan, X_plan_t, Y_plan_t, "Traj")
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = run(ObjModel(), X_plan, Y_plan, X_plan_t, Y_plan_t, "Obj")
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

# Test on standard prediction
print("\n--- Transfer: Planning → Prediction ---")
mse, rnd = run(TrajModel(), X_plan, Y_plan, X_pred_t, Y_pred_t, "Traj")
print(f"Traj: {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")
mse, rnd = run(ObjModel(), X_plan, Y_plan, X_pred_t, Y_pred_t, "Obj")
print(f"Obj:  {mse:.4f} ({(rnd-mse)/rnd*100:.0f}%)")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("\nPlanning requires:")
print("  1. Understanding goal-directed behavior")
print("  2. Trajectory optimization")
print("  3. World model")
print("\nIf object representation helps:")
print("  → Better planning accuracy")
print("\nIf not:")
print("  → Both models struggle equally")
