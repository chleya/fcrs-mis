"""
VISUALIZATION: Trajectory vs Object Model Predictions
Compare 6-object scaling predictions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MAX_OBJ = 6

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

def generate_test_cases(n=6):
    """Generate 6 test cases with different object counts"""
    cases = []
    for n_obj in [2, 3, 4, 5, 6, 6]:
        random.seed(100 + n_obj)
        np.random.seed(100 + n_obj)
        
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(n_obj)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(n_obj)]
        
        # t0 features
        f0 = make_features(pos, vel)
        
        # Save initial
        init_state = [(x, y, vx, vy) for (x, y), (vx, vy) in zip(pos, vel)]
        
        # Simulate 10 steps
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]
                vx, vy = vel[i]
                x, y = x + vx*0.5, y + vy*0.5
                if x < 3 or x > 29: vx *= -1
                if y < 3 or y > 29: vy *= -1
                pos[i], vel[i] = (x, y), (vx, vy)
        
        # t10 features
        f10 = make_features(pos, vel)
        
        cases.append({
            'n_objects': n_obj,
            'input': np.array(f0 + f10, dtype=np.float32),
            'true_pos': pos[0][0] / 32
        })
    
    return cases

class TrajModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.fc(x).squeeze()

class ObjModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16))
        self.out = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 2, MAX_OBJ, 4)
        h = self.proc(x[:, 1])
        return self.out(h[:, 0]).squeeze()

# Generate data and train
def generate_data(n, n_objects, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

print("Training models...")
X_tr, Y_tr = generate_data(2000, 2, 42)
X_tr = torch.FloatTensor(X_tr)
Y_tr = torch.FloatTensor(Y_tr)

# Train Trajectory
m_traj = TrajModel()
opt = torch.optim.Adam(m_traj.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_tr))
    for i in range(0, len(X_tr), 64):
        p = m_traj(X_tr[idx[i:i+64]])
        loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

# Train Object
m_obj = ObjModel()
opt = torch.optim.Adam(m_obj.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X_tr))
    for i in range(0, len(X_tr), 64):
        p = m_obj(X_tr[idx[i:i+64]])
        loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()

# Generate test cases
test_cases = generate_test_cases(6)

# Test
m_traj.eval()
m_obj.eval()

print("\n" + "="*60)
print("PREDICTION COMPARISON (6-object test)")
print("="*60)

results = []
for case in test_cases:
    x = torch.FloatTensor(case['input']).unsqueeze(0)
    pred_traj = m_traj(x).item()
    pred_obj = m_obj(x).item()
    true = case['true_pos']
    
    results.append({
        'n_objects': case['n_objects'],
        'traj': pred_traj,
        'obj': pred_obj,
        'true': true,
        'traj_err': abs(pred_traj - true),
        'obj_err': abs(pred_obj - true)
    })
    
    print(f"\n{case['n_objects']} objects:")
    print(f"  True:     {true:.3f}")
    print(f"  Traj:     {pred_traj:.3f} (err: {abs(pred_traj-true):.3f})")
    print(f"  Object:   {pred_obj:.3f} (err: {abs(pred_obj-true):.3f})")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, case in enumerate(test_cases):
    ax = axes[idx // 3, idx % 3]
    
    n_obj = case['n_objects']
    traj_err = results[idx]['traj_err']
    obj_err = results[idx]['obj_err']
    
    # Bar chart
    models = ['Trajectory', 'Object']
    errors = [traj_err, obj_err]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(models, errors, color=colors, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'{n_obj} Objects')
    ax.set_ylim(0, max(traj_err, obj_err) * 1.2)
    
    # Add value labels
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{err:.3f}', ha='center', va='bottom', fontsize=10)

plt.suptitle('Trajectory vs Object Model: Prediction Errors by Object Count', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('F:/fcrs_mis/prediction_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: F:/fcrs_mis/prediction_comparison.png")

# Summary plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

obj_counts = [r['n_objects'] for r in results]
traj_errors = [r['traj_err'] for r in results]
obj_errors = [r['obj_err'] for r in results]

x = np.arange(len(obj_counts))
width = 0.35

bars1 = ax2.bar(x - width/2, traj_errors, width, label='Trajectory', color='#FF6B6B')
bars2 = ax2.bar(x + width/2, obj_errors, width, label='Object', color='#4ECDC4')

ax2.set_xlabel('Number of Objects')
ax2.set_ylabel('Prediction Error')
ax2.set_title('Combinatorial Scaling: Trajectory vs Object Model')
ax2.set_xticks(x)
ax2.set_xticklabels(obj_counts)
ax2.legend()

# Add training line
ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Training (2 objects)')

plt.tight_layout()
plt.savefig('F:/fcrs_mis/scaling_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: F:/fcrs_mis/scaling_comparison.png")

# Summary stats
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
avg_traj = np.mean(traj_errors[1:])  # Exclude 2 objects (training)
avg_obj = np.mean(obj_errors[1:])

print(f"Average error (3-6 objects):")
print(f"  Trajectory: {avg_traj:.4f}")
print(f"  Object:     {avg_obj:.4f}")
print(f"  Ratio:     {avg_traj/avg_obj:.1f}x")

print("\nDone!")
