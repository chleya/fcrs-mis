"""
Two Bouncing Balls - HARD VERSION
More complexity: random colors, sizes, backgrounds
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# 1. PHYSICS ENGINE - Complex Two Bouncing Balls
# ============================================================

class TwoBallsHard:
    def __init__(self, size=32):
        self.size = size
        
    def reset(self):
        # Random colors for each ball
        self.color1 = np.random.rand(3)
        self.color2 = np.random.rand(3)
        
        # Random sizes
        self.radius1 = random.randint(2, 5)
        self.radius2 = random.randint(2, 5)
        
        # Random positions
        self.ball1 = np.array([
            np.random.randint(5, self.size - 5),
            np.random.randint(5, self.size - 5)
        ], dtype=np.float32)
        self.ball2 = np.array([
            np.random.randint(5, self.size - 5),
            np.random.randint(5, self.size - 5)
        ], dtype=np.float32)
        
        # Random velocities
        self.vel1 = np.random.randn(2).astype(np.float32) * 3
        self.vel2 = np.random.randn(2).astype(np.float32) * 3
        
        # Random background color
        self.bg_color = np.random.rand(3) * 0.3
        
    def step(self):
        self.ball1 += self.vel1 * 0.5
        self.ball2 += self.vel2 * 0.5
        
        # Bounce
        for ball, vel, r in [(self.ball1, self.vel1, self.radius1), 
                              (self.ball2, self.vel2, self.radius2)]:
            for i in range(2):
                if ball[i] < r:
                    ball[i] = r
                    vel[i] = abs(vel[i])
                elif ball[i] > self.size - r:
                    ball[i] = self.size - r
                    vel[i] = -abs(vel[i])
        
        # Collision
        dist = np.linalg.norm(self.ball1 - self.ball2)
        min_dist = self.radius1 + self.radius2
        if dist < min_dist and dist > 0:
            normal = (self.ball1 - self.ball2) / dist
            rel_vel = self.vel1 - self.vel2
            impulse = np.dot(rel_vel, normal)
            self.vel1 -= impulse * normal
            self.vel2 += impulse * normal
            
    def render(self):
        img = np.ones((self.size, self.size, 3), dtype=np.float32) * self.bg_color
        
        # Ball 1
        for dx in range(-self.radius1, self.radius1+1):
            for dy in range(-self.radius1, self.radius1+1):
                if dx*dx + dy*dy <= self.radius1*self.radius1:
                    x, y = int(self.ball1[0]+dx), int(self.ball1[1]+dy)
                    if 0 <= x < self.size and 0 <= y < self.size:
                        img[y, x] = self.color1
                        
        # Ball 2
        for dx in range(-self.radius2, self.radius2+1):
            for dy in range(-self.radius2, self.radius2+1):
                if dx*dx + dy*dy <= self.radius2*self.radius2:
                    x, y = int(self.ball2[0]+dx), int(self.ball2[1]+dy)
                    if 0 <= x < self.size and 0 <= y < self.size:
                        img[y, x] = self.color2
                        
        return img

# ============================================================
# 2. MODELS
# ============================================================

class Baseline(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        z = self.fc_enc(h)
        h = self.fc_dec(z).reshape(-1, 128, 4, 4)
        return self.decoder(h), z

# ============================================================
# 3. MAIN
# ============================================================

def generate_data(n_seqs=2000, seq_len=10, size=32):
    env = TwoBallsHard(size)
    sequences = []
    
    for _ in range(n_seqs):
        env.reset()
        seq = []
        for _ in range(seq_len):
            seq.append(env.render())
            env.step()
        sequences.append(np.array(seq))
        
    return np.array(sequences)

print("="*60)
print("HARD OBJECT EMERGENCE EXPERIMENT")
print("="*60)

# Generate data
print("\n1. Generating complex dataset...")
train_seqs = generate_data(n_seqs=2000, seq_len=10, size=32)
test_seqs = generate_data(n_seqs=500, seq_len=10, size=32)

train_data = torch.FloatTensor(train_seqs).permute(0, 1, 4, 2, 3) / 1.0
test_data = torch.FloatTensor(test_seqs).permute(0, 1, 4, 2, 3) / 1.0

print(f"Train: {train_data.shape}")
print(f"Test: {test_data.shape}")

# Train
print("\n2. Training Baseline (dim=16)...")
model = Baseline(16)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(5):
    idx = torch.randperm(len(train_data))
    losses = []
    
    for i in range(0, len(idx), 32):
        batch = train_data[idx[i:i+32]]
        pred, z = model(batch[:, 0])
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
    
print(f"\n3. Test MSE: {mse:.4f}")

# Check if model is learning something meaningful
print(f"\n4. Latent stats:")
print(f"  z mean: {z.mean().item():.4f}")
print(f"  z std: {z.std().item():.4f}")

# Show some predictions
print("\n5. Sample predictions (first frame):")
print(f"  True: {test_data[0, 1, 0, 10, 10].item():.3f}")
print(f"  Pred: {pred[0, 0, 10, 10].item():.3f}")
