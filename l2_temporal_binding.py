"""
L2 MINIMAL EXPERIMENT: Temporal Binding Constraint

Core idea: 
- slot_i(t+1) ≈ predict(slot_i(t))
- Force slots to maintain persistent object state across time

This is the ONE decisive experiment that could close the L2 loop.
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
print("L2 MINIMAL: TEMPORAL BINDING CONSTRAINT")
print("="*60)

def generate_data(n=3000):
    """Generate identity tracking with multiple frames for temporal binding"""
    X_frames = []  # t0, t2, t4, t6, t8, t10
    Y_a = []  # Position of ball A at t10
    
    for _ in range(n):
        # t0: colored balls
        x_a, y = random.uniform(5, 12), random.uniform(10, 22)
        x_b = random.uniform(20, 27)
        
        vx_a, vx_b = random.uniform(-2, 2), random.uniform(-2, 2)
        
        frames = []
        
        # Generate frames at intervals
        for t in [0, 2, 4, 6, 8, 10]:
            if t == 0:
                # t0: colored
                img = np.zeros((32, 32, 3), np.float32)
                img[int(y), int(x_a)] = [1, 0, 0]  # Red = A
                img[int(y), int(x_b)] = [0, 0, 1]  # Blue = B
            else:
                # Move
                for _ in range(2):  # 2 steps per frame
                    x_a += vx_a * 0.5
                    x_b += vx_b * 0.5
                    if x_a < 3 or x_a > 29: vx_a *= -1
                
                # t10: teleport (break motion inference)
                if t == 10:
                    x_a = random.uniform(5, 27)
                    x_b = random.uniform(5, 27)
                
                # Clamp positions
                x_a = max(3, min(28, x_a))
                x_b = max(3, min(28, x_b))
                
                # White balls
                img = np.zeros((32, 32, 3), np.float32)
                img[int(y), int(x_a)] = [1, 1, 1]
                img[int(y), int(x_b)] = [1, 1, 1]
            
            frames.append(img)
        
        X_frames.append(frames)
        Y_a.append(x_a / 32)
    
    return np.array(X_frames), np.array(Y_a)

print("\n1. Generating data...")
X, Y = generate_data(3000)
# X: (N, 6, 32, 32, 3)
X = torch.FloatTensor(X).permute(0, 1, 4, 2, 3)  # (N, 6, 3, 32, 32)
Y = torch.FloatTensor(Y)

print(f"   X: {X.shape}, Y: {Y.shape}")

# ========== Model 1: Baseline ==========
class Baseline(nn.Module):
    """Standard CNN over all frames"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8*6, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # x: (B, 6, 3, 32, 32)
        B, T = x.size(0), x.size(1)
        x = x.reshape(B * T, 3, 32, 32)  # (B*6, 3, 32, 32)
        h = self.enc(x).flatten(1)  # (B*6, 64*8*8)
        h = h.reshape(B, -1)  # (B, 64*8*8*6)
        return self.fc(h).squeeze()

# ========== Model 2: Temporal Binding Slot ==========
class TemporalBindingSlot(nn.Module):
    """
    Key innovation: temporal binding constraint
    - slot_i(t+1) = dynamics(slot_i(t))
    - Forces persistent object state across time
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 64), nn.ReLU(),
        )
        
        # Two slots for two objects
        self.slot = nn.Parameter(torch.randn(2, 64) * 0.1)
        
        # Dynamics model: predicts next slot state
        self.dynamics = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Predictor for final position
        self.predictor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, return_temporal_loss=False):
        """
        x: (B, T, 3, 32, 32)
        """
        B, T = x.size(0), x.size(1)
        
        # Encode each frame
        x_flat = x.view(B*T, 3, 32, 32)
        h = self.encoder(x_flat)  # (B*T, 64)
        h = h.view(B, T, -1)  # (B, T, 64)
        
        # Add slots to each frame
        h = h.unsqueeze(2) + self.slot.unsqueeze(0).unsqueeze(1)  # (B, T, 2, 64)
        
        # Temporal binding loss
        temporal_loss = 0
        for t in range(T-1):
            # Current slot states
            slot_t = h[:, t, :, :]  # (B, 2, 64)
            slot_t1 = h[:, t+1, :, :,]
            
            # Predict next state
            pred_t1 = self.dynamics(slot_t)  # (B, 2, 64)
            
            # Binding loss: predict should match actual
            temporal_loss += F.mse_loss(pred_t1, slot_t1)
        
        # Use final frame to predict
        final_slot = h[:, -1, 0, :]  # Slot 0 for object A
        pred = self.predictor(final_slot).squeeze()
        
        if return_temporal_loss:
            return pred, temporal_loss
        return pred

print("\n2. Training Baseline...")
m = Baseline()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(10):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 64):
        p = m(X[idx[i:i+64]])
        loss = F.mse_loss(p, Y[idx[i:i+64]])
        opt.zero_grad(); loss.backward(); opt.step()
mse_baseline = F.mse_loss(m(X), Y).item()

print("3. Training Temporal Binding Slot...")
m = TemporalBindingSlot()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for ep in range(10):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 64):
        p, temp_loss = m(X[idx[i:i+64]], return_temporal_loss=True)
        
        # Main prediction loss
        loss_main = F.mse_loss(p, Y[idx[i:i+64]])
        
        # Temporal binding loss
        loss = loss_main + 0.1 * temp_loss
        
        opt.zero_grad(); loss.backward(); opt.step()

pred, temp_loss = m(X, return_temporal_loss=True)
mse_binding = F.mse_loss(pred, Y).item()

random_mse = Y.var().item()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Baseline:            MSE = {mse_baseline:.4f} ({(random_mse-mse_baseline)/random_mse*100:.1f}% < random)")
print(f"Temporal Binding:    MSE = {mse_binding:.4f} ({(random_mse-mse_binding)/random_mse*100:.1f}% < random)")
print(f"Random:             MSE = {random_mse:.4f}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
if mse_binding < mse_baseline * 0.5:
    print("=> TEMPORAL BINDING WORKS!")
    print("=> Structure → Identity capability verified")
elif mse_binding < mse_baseline:
    print("=> Temporal binding helps somewhat")
else:
    print("=> Temporal binding not sufficient")
    print("=> Need stronger structure or different approach")
