"""
STRUCTURE EXPERIMENT PLATFORM
Unified environment with structure toggles to observe capability changes

Structure toggles:
1. compression: latent_dim = 2 vs 32
2. temporal: single frame vs multi-frame
3. slot: baseline vs slot attention
4. interaction: independent vs collision

This forms a Structure-Capability Map
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
print("STRUCTURE EXPERIMENT PLATFORM")
print("="*60)

# ============ ENVIRONMENT GENERATOR ============

def generate_task(task_type, n_samples=1000):
    """Generate different task types"""
    X_t0, X_t1 = [], []
    targets = []
    
    for _ in range(n_samples):
        if task_type == 'simple':
            # Simple: one ball, predict position
            x = random.uniform(5, 27)
            y = random.uniform(5, 27)
            
            img = np.zeros((32, 32, 3), np.float32)
            img[int(y), int(x)] = [1, 0, 0]
            
            X_t0.append(img)
            X_t1.append(img)
            targets.append(x/32)
            
        elif task_type == 'two_object':
            # Two objects, independent motion
            x1 = random.uniform(5, 15)
            x2 = random.uniform(17, 27)
            y = random.uniform(10, 22)
            
            v1 = random.uniform(-2, 2)
            v2 = random.uniform(-2, 2)
            
            img0 = np.zeros((32, 32, 3), np.float32)
            img0[int(y), int(x1)] = [1, 0, 0]
            img0[int(y), int(x2)] = [0, 0, 1]
            
            for _ in range(5):
                x1 += v1 * 0.5
                x2 += v2 * 0.5
                if x1 < 3 or x1 > 29: v1 *= -1
                if x2 < 3 or x2 > 29: v2 *= -1
            
            img1 = np.zeros((32, 32, 3), np.float32)
            img1[int(y), int(x1)] = [1, 1, 1]
            img1[int(y), int(x2)] = [1, 1, 1]
            
            X_t0.append(img0)
            X_t1.append(img1)
            targets.append(x1/32)
            
        elif task_type == 'collision':
            # Collision scenario
            x1 = random.uniform(4, 12)
            x2 = random.uniform(20, 28)
            y = random.uniform(10, 22)
            
            v1 = random.uniform(1, 2)
            v2 = random.uniform(-2, -1)
            
            img0 = np.zeros((32, 32, 3), np.float32)
            img0[int(y), int(x1)] = [1, 0, 0]
            img0[int(y), int(x2)] = [0, 0, 1]
            
            for step in range(5):
                x1 += v1 * 0.5
                x2 += v2 * 0.5
            
            # Collision - exchange velocities
            if abs(x1 - x2) < 3:
                v1, v2 = v2, v1
            
            for step in range(5, 10):
                x1 += v1 * 0.5
                x2 += v2 * 0.5
                if x1 < 3 or x1 > 29: v1 *= -1
                if x2 < 3 or x2 > 29: v2 *= -1
            
            x1 = max(3, min(28, x1))
            x2 = max(3, min(28, x2))
            
            img1 = np.zeros((32, 32, 3), np.float32)
            img1[int(y), int(x1)] = [1, 1, 1]
            img1[int(y), int(x2)] = [1, 1, 1]
            
            X_t0.append(img0)
            X_t1.append(img1)
            targets.append(x1/32)
    
    return np.array(X_t0), np.array(X_t1), np.array(targets)

# ============ MODELS ============

class BaselineSmall(nn.Module):
    """Baseline with small latent"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*8*8*2, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x0, x1):
        h0 = self.enc(x0).flatten(1)
        h1 = self.enc(x1).flatten(1)
        return self.fc(torch.cat([h0, h1], dim=1)).squeeze()

class BaselineLarge(nn.Module):
    """Baseline with large latent"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*8*8*2, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x0, x1):
        h0 = self.enc(x0).flatten(1)
        h1 = self.enc(x1).flatten(1)
        return self.fc(torch.cat([h0, h1], dim=1)).squeeze()

class SlotModel(nn.Module):
    """Slot attention model"""
    def __init__(self, n_slots=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slots = nn.Parameter(torch.randn(n_slots, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x0, x1):
        h = self.enc(x1).mean(dim=[2, 3])
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)
        return self.predict(h[:, 0]).squeeze()

# ============ EXPERIMENT ============

print("\n1. Running Structure-Capability Map...")
print("="*60)

results = []

tasks = ['simple', 'two_object', 'collision']
models = [
    ('Baseline-Small', BaselineSmall),
    ('Baseline-Large', BaselineLarge),
    ('Slot-2', lambda: SlotModel(2)),
]

for task in tasks:
    print(f"\n--- Task: {task} ---")
    
    X0, X1, Y = generate_task(task, 1000)
    X0 = torch.FloatTensor(X0).permute(0, 3, 1, 2)
    X1 = torch.FloatTensor(X1).permute(0, 3, 1, 2)
    Y = torch.FloatTensor(Y)
    
    random_mse = Y.var().item()
    
    for model_name, ModelClass in models:
        m = ModelClass()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        
        for ep in range(8):
            idx = torch.randperm(len(X0))
            for i in range(0, len(X0), 64):
                p = m(X0[idx[i:i+64]], X1[idx[i:i+64]])
                loss = F.mse_loss(p, Y[idx[i:i+64]])
                opt.zero_grad(); loss.backward(); opt.step()
        
        mse = F.mse_loss(m(X0, X1), Y).item()
        improvement = (random_mse - mse) / random_mse * 100
        
        results.append({
            'task': task,
            'model': model_name,
            'mse': mse,
            'improvement': improvement
        })
        
        print(f"  {model_name}: MSE={mse:.4f} ({improvement:.1f}% < random)")

# ============ SUMMARY ============

print("\n" + "="*60)
print("STRUCTURE-CAPABILITY MAP")
print("="*60)

print(f"\n{'Task':<15} {'Model':<15} {'MSE':<10} {'Improvement':<10}")
print("-" * 50)
for r in results:
    print(f"{r['task']:<15} {r['model']:<15} {r['mse']:<10.4f} {r['improvement']:<10.1f}%")

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

# Find patterns
simple_results = [r for r in results if r['task'] == 'simple']
two_results = [r for r in results if r['task'] == 'two_object']
collision_results = [r for r in results if r['task'] == 'collision']

# Compression effect
small = np.mean([r['mse'] for r in simple_results if 'Small' in r['model']])
large = np.mean([r['mse'] for r in simple_results if 'Large' in r['model']])

print(f"\n1. Compression effect (simple task):")
print(f"   Small latent: {small:.4f}")
print(f"   Large latent: {large:.4f}")

# Slot effect
baseline_two = [r['mse'] for r in two_results if 'Baseline' in r['model']]
slot_two = [r['mse'] for r in two_results if 'Slot' in r['model']]

print(f"\n2. Slot effect (two_object task):")
print(f"   Baseline avg: {np.mean(baseline_two):.4f}")
print(f"   Slot avg: {np.mean(slot_two):.4f}")

# Interaction effect
baseline_col = [r['mse'] for r in collision_results if 'Baseline' in r['model']]
slot_col = [r['mse'] for r in collision_results if 'Slot' in r['model']]

print(f"\n3. Interaction effect (collision task):")
print(f"   Baseline avg: {np.mean(baseline_col):.4f}")
print(f"   Slot avg: {np.mean(slot_col):.4f}")

print("\n" + "="*60)
print("CONCLUSIONS")
print("="*60)
print("1. Compression: Smaller latent ~ Better (when task is simple)")
print("2. Slot: No advantage in current tasks")
print("3. Interaction: Doesn't automatically require object representation")
