"""
Two Bouncing Balls - Object Emergence Experiment
Minimal version for quick testing
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# 1. PHYSICS ENGINE - Two Bouncing Balls
# ============================================================

class TwoBalls:
    def __init__(self, size=32):
        self.size = size
        self.radius = 3
        
    def reset(self):
        # Random positions (avoid overlap)
        self.ball1 = np.array([
            np.random.randint(5, self.size//2 - 5),
            np.random.randint(5, self.size - 5)
        ], dtype=np.float32)
        self.ball2 = np.array([
            np.random.randint(self.size//2 + 5, self.size - 5),
            np.random.randint(5, self.size - 5)
        ], dtype=np.float32)
        
        # Random velocities
        self.vel1 = np.random.randn(2).astype(np.float32) * 2
        self.vel2 = np.random.randn(2).astype(np.float32) * 2
        
    def step(self):
        # Update positions
        self.ball1 += self.vel1 * 0.5
        self.ball2 += self.vel2 * 0.5
        
        # Bounce off walls
        for ball, vel in [(self.ball1, self.vel1), (self.ball2, self.vel2)]:
            for i in range(2):
                if ball[i] < self.radius:
                    ball[i] = self.radius
                    vel[i] = abs(vel[i])
                elif ball[i] > self.size - self.radius:
                    ball[i] = self.size - self.radius
                    vel[i] = -abs(vel[i])
        
        # Ball-ball collision (simple)
        dist = np.linalg.norm(self.ball1 - self.ball2)
        if dist < self.radius * 2:
            normal = (self.ball1 - self.ball2) / (dist + 1e-6)
            rel_vel = self.vel1 - self.vel2
            impulse = np.dot(rel_vel, normal)
            self.vel1 -= impulse * normal
            self.vel2 += impulse * normal
            
    def render(self):
        img = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # Draw ball1 (red)
        for dx in range(-self.radius, self.radius+1):
            for dy in range(-self.radius, self.radius+1):
                if dx*dx + dy*dy <= self.radius*self.radius:
                    x, y = int(self.ball1[0]+dx), int(self.ball1[1]+dy)
                    if 0 <= x < self.size and 0 <= y < self.size:
                        img[y, x] = [1, 0, 0]
                        
        # Draw ball2 (blue)
        for dx in range(-self.radius, self.radius+1):
            for dy in range(-self.radius, self.radius+1):
                if dx*dx + dy*dy <= self.radius*self.radius:
                    x, y = int(self.ball2[0]+dx), int(self.ball2[1]+dy)
                    if 0 <= x < self.size and 0 <= y < self.size:
                        img[y, x] = [0, 0, 1]
                        
        return img

# ============================================================
# 2. DATASET GENERATOR
# ============================================================

def generate_sequences(n_seqs=1000, seq_len=20, size=32):
    env = TwoBalls(size)
    sequences = []
    
    for _ in range(n_seqs):
        env.reset()
        seq = []
        for _ in range(seq_len):
            img = env.render()
            seq.append(img)
            env.step()
        sequences.append(np.array(seq))
        
    return np.array(sequences)

# ============================================================
# 3. MODELS
# ============================================================

class Baseline(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        z = self.fc_enc(h)
        h = self.fc_dec(z).reshape(-1, 128, 4, 4)
        out = self.decoder(h)
        return out, z

class CapacityModel(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        z = self.fc_enc(h)
        h = self.fc_dec(z).reshape(-1, 128, 4, 4)
        out = self.decoder(h)
        return out, z

# ============================================================
# 4. MAIN
# ============================================================

print("="*60)
print("OBJECT EMERGENCE EXPERIMENT")
print("="*60)

# Generate data
print("\n1. Generating dataset...")
train_seqs = generate_sequences(n_seqs=3000, seq_len=10, size=32)
test_seqs = generate_sequences(n_seqs=500, seq_len=10, size=32)

# Prepare tensors
train_data = torch.FloatTensor(train_seqs).permute(0, 1, 4, 2, 3) / 255.0
test_data = torch.FloatTensor(test_seqs).permute(0, 1, 4, 2, 3) / 255.0

print(f"Train: {train_data.shape}")
print(f"Test: {test_data.shape}")

# Train models
print("\n2. Training models...")

models = {
    'Baseline (dim=16)': Baseline(16),
    'Capacity (dim=4)': CapacityModel(4),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Train for few epochs
    for epoch in range(3):
        idx = torch.randperm(len(train_data))
        losses = []
        
        for i in range(0, len(idx), 32):
            batch = train_data[idx[i:i+32]]
            pred, _ = model(batch[:, 0])
            loss = F.mse_loss(pred, batch[:, 1])
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            
        print(f"  Epoch {epoch+1}: loss={np.mean(losses):.4f}")
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred, z = model(test_data[:, 0])
        mse = F.mse_loss(pred, test_data[:, 1]).item()
        
    results[name] = {'mse': mse}
    print(f"  Test MSE: {mse:.4f}")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for name, r in results.items():
    print(f"{name}: MSE={r['mse']:.4f}")

print("\nNOTE: Full slot alignment evaluation requires")
print("additional tracking of ball positions.")
