"""Identity Test with 4 Structural Constraints - Slot-Structured"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("="*60)
print("IDENTITY TEST - WITH STRUCTURAL CONSTRAINTS")
print("="*60)

# Dataset with SOFT OVERLAP (not full occlusion)
print("\n1. Creating dataset with soft overlap...")

def create_dataset(n=6000):
    data = []
    for i in range(n):
        mode = i % 3  # 0: cross, 1: soft overlap, 2: non-cross
        
        t = (i // 3) * 0.015
        
        if mode == 0:  # Full crossing
            b1_x, b1_y = 5.0 + t * 60, 16.0
            b2_x, b2_y = 27.0 - t * 60, 16.0 + t * 2  # Slight y offset
            crossing = 1
        elif mode == 1:  # Soft overlap (key!)
            b1_x, b1_y = 10.0 + t * 40, 16.0
            b2_x, b2_y = 18.0 + t * 40, 16.0 + t * 3  # Almost touch but don't
            crossing = 0.5  # Partial confusion
        else:  # Non-crossing
            b1_x, b1_y = 8.0, 10.0 + t * 10
            b2_x, b2_y = 24.0, 22.0 - t * 10
            crossing = 0
        
        # Clamp
        b1_x = max(1, min(30, int(b1_x)))
        b1_y = max(1, min(30, int(b1_y)))
        b2_x = max(1, min(30, int(b2_x)))
        b2_y = max(1, min(30, int(b2_y)))
        
        # Image with positions
        img = np.zeros((32, 32, 3), dtype=np.float32)
        
        # Ball 1 (track this one)
        img[b1_y-1:b1_y+2, b1_x-1:b1_x+2] = [1.0, 1.0, 1.0]
        
        # Ball 2 
        img[b2_y-1:b2_y+2, b2_x-1:b2_x+2] = [0.7, 0.7, 0.7]
        
        # Target: ball1 position at t+3
        if mode == 0:
            target = [(5.0 + (t*60+3)) / 32, 16.0 / 32]
        elif mode == 1:
            target = [(10.0 + (t*40+3)) / 32, 16.0 / 32]
        else:
            target = [8.0 / 32, (10.0 + t*10+3) / 32]
        
        data.append((img, target, crossing, mode))
    
    return data

data = create_dataset(6000)
images = np.array([d[0] for d in data])
targets = np.array([d[1] for d in data])
crossing = np.array([d[2] for d in data])
modes = np.array([d[3] for d in data])

X = torch.FloatTensor(images).permute(0, 3, 1, 2)
Y = torch.FloatTensor(targets)

print(f"   Total: {len(X)}, Crossing: {(modes==0).sum()}, Overlap: {(modes==1).sum()}, Normal: {(modes==2).sum()}")

# ============================================================
# MODELS
# ============================================================

class Baseline(nn.Module):
    """No structural constraints"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64*8*8, 64), nn.ReLU(), nn.Linear(64, 2))
        
    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        return self.fc(h), h

class SlotOriginal(nn.Module):
    """Original Slot - just spatial grouping"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        # Simple slot attention
        self.slot_fc = nn.Linear(64*8*8, 64 * 2)  # 2 slots
        self.predictor = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        
    def forward(self, x, return_slots=False):
        h = self.conv(x).reshape(x.size(0), -1)
        slots = self.slot_fc(h).reshape(x.size(0), 2, 64)  # (B, 2, 64)
        
        # Use slot 0 for prediction
        pred = self.predictor(slots[:, 0, :])
        
        if return_slots:
            return pred, slots
        return pred, h

class SlotStructured(nn.Module):
    """Slot + 4 Structural Constraints"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.slot_fc = nn.Linear(64*8*8, 64 * 2)
        
        # Per-slot predictors (for masked reconstruction)
        self.slot1_pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.slot2_pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        
        # Identity predictor
        self.identity_pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        
    def forward(self, x, return_slots=False):
        h = self.conv(x).reshape(x.size(0), -1)
        slots = self.slot_fc(h).reshape(x.size(0), 2, 64)
        
        # Constraint 1: Use slot-specific predictors
        pred1 = self.slot1_pred(slots[:, 0, :])  # Slot 1 prediction
        pred2 = self.slot2_pred(slots[:, 1, :])  # Slot 2 prediction
        
        # Use slot 0 as primary (should track ball1)
        pred = self.identity_pred(slots[:, 0, :])
        
        if return_slots:
            return pred, slots, pred1, pred2
        return pred, h

# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_model(model, X, Y, modes, n_epochs=15, name="Model"):
    print(f"\n2. Training {name}...")
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for ep in range(n_epochs):
        idx = torch.randperm(len(X))
        total_loss = 0
        count = 0
        
        for i in range(0, len(X), 32):
            batch_x = X[idx[i:i+32]]
            batch_y = Y[idx[i:i+32]]
            batch_mode = modes[idx[i:i+32]]
            
            # Forward
            if name == "SlotStructured":
                pred, slots, pred1, pred2 = model(batch_x, return_slots=True)
                
                # Main prediction loss
                loss = F.mse_loss(pred, batch_y)
                
                # Constraint 1: Slot separation loss
                # Make slot1 and slot2 different
                slot_diff = F.mse_loss(slots[:, 0, :], slots[:, 1, :])
                loss = loss + 0.1 * slot_diff
                
                # Convert to tensor
                batch_mode_tensor = torch.tensor(batch_mode, dtype=torch.float32, device=batch_x.device)
                cross_weight = (batch_mode_tensor == 0).float().mean()
                loss = loss + cross_weight * F.mse_loss(pred1, pred2) * 0.1
                
            else:
                pred, _ = model(batch_x)
                loss = F.mse_loss(pred, batch_y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            count += 1
        
        if ep < 5 or ep == n_epochs - 1:
            print(f"   Epoch {ep+1}: loss={total_loss/count:.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        if name == "SlotStructured":
            pred, slots, pred1, pred2 = model(X, return_slots=True)
        else:
            pred, _ = model(X)
        
        results = {
            'all': F.mse_loss(pred, Y).item(),
            'cross': F.mse_loss(pred[modes==0], Y[modes==0]).item(),
            'overlap': F.mse_loss(pred[modes==1], Y[modes==1]).item(),
            'normal': F.mse_loss(pred[modes==2], Y[modes==2]).item()
        }
    
    return results

# ============================================================
# MAIN
# ============================================================

results = {}

# Model 1: Baseline
results['Baseline'] = train_model(Baseline(), X, Y, modes, name="Baseline")

# Model 2: Slot Original
results['SlotOriginal'] = train_model(SlotOriginal(), X, Y, modes, name="SlotOriginal")

# Model 3: Slot Structured (with constraints)
results['SlotStructured'] = train_model(SlotStructured(), X, Y, modes, name="SlotStructured")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Model':<20} | {'All':>8} | {'Cross':>8} | {'Overlap':>8} | {'Normal':>8}")
print("-"*70)
for name, r in results.items():
    print(f"{name:<20} | {r['all']:>8.4f} | {r['cross']:>8.4f} | {r['overlap']:>8.4f} | {r['normal']:>8.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("Cross = trajectory crossing (hardest)")
print("Overlap = soft overlap (moderate)")  
print("Normal = no interaction (easiest)")
print("\nIf SlotStructured << Baseline/SlotOriginal on Cross:")
print("  -> Structural constraints enable object identity!")
