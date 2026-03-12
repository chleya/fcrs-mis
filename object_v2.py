"""
Object Emergence - Better Slot Model with Per-Slot Decoding
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
# PHYSICS ENGINE
# ============================================================

class TwoBallsHard:
    def __init__(self, size=32):
        self.size = size
        
    def reset(self):
        self.color1 = np.random.rand(3)
        self.color2 = np.random.rand(3)
        self.radius1 = random.randint(2, 5)
        self.radius2 = random.randint(2, 5)
        self.ball1 = np.array([np.random.randint(5, self.size-5), np.random.randint(5, self.size-5)], dtype=np.float32)
        self.ball2 = np.array([np.random.randint(5, self.size-5), np.random.randint(5, self.size-5)], dtype=np.float32)
        self.vel1 = np.random.randn(2).astype(np.float32) * 3
        self.vel2 = np.random.randn(2).astype(np.float32) * 3
        self.bg_color = np.random.rand(3) * 0.3
        
    def step(self):
        self.ball1 += self.vel1 * 0.5
        self.ball2 += self.vel2 * 0.5
        for ball, vel, r in [(self.ball1, self.vel1, self.radius1), (self.ball2, self.vel2, self.radius2)]:
            for i in range(2):
                if ball[i] < r: ball[i], vel[i] = r, abs(vel[i])
                elif ball[i] > self.size-r: ball[i], vel[i] = self.size-r, -abs(vel[i])
        dist = np.linalg.norm(self.ball1 - self.ball2)
        if dist < self.radius1 + self.radius2 and dist > 0:
            normal = (self.ball1 - self.ball2) / dist
            impulse = np.dot(self.vel1 - self.vel2, normal)
            self.vel1 -= impulse * normal
            self.vel2 += impulse * normal
            
    def render(self):
        img = np.ones((self.size, self.size, 3), dtype=np.float32) * self.bg_color
        for ball, color, r in [(self.ball1, self.color1, self.radius1), (self.ball2, self.color2, self.radius2)]:
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        x, y = int(ball[0]+dx), int(ball[1]+dy)
                        if 0 <= x < self.size and 0 <= y < self.size:
                            img[y, x] = color
        return img

# ============================================================
# MODELS
# ============================================================

class Baseline(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        self.fc_enc = nn.Linear(128*4*4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        
    def forward(self, x):
        h = self.encoder(x).reshape(x.size(0), -1)
        z = self.fc_enc(h)
        h = self.fc_dec(z).reshape(-1, 128, 4, 4)
        return self.decoder(h), z

class SlotModelV2(nn.Module):
    """Slot model with per-slot decoding and masking"""
    def __init__(self, n_slots=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        
        # Slot attention
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)
        self.mlp = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        
        # Per-slot decoder
        self.slot_dec = nn.Linear(128, 128*4*4)
        
        # Final decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        
    def forward(self, x):
        h = self.encoder(x)  # (B, 128, 4, 4)
        h_flat = h.flatten(2).transpose(1, 2)  # (B, 16, 128)
        
        # Slot attention
        slots = torch.randn(x.size(0), 2, 128, device=x.device)
        for _ in range(3):
            q = self.query(slots)
            k = self.key(h_flat).mean(1, keepdim=True)
            v = self.value(h_flat).mean(1, keepdim=True)
            attn = torch.softmax(q @ k.transpose(-2,-1) / 8, -1)
            slots = attn @ v + slots
            slots = slots + self.mlp(slots)
        
        # Decode each slot and sum
        outputs = []
        for i in range(2):
            slot_h = self.slot_dec(slots[:, i]).reshape(-1, 128, 4, 4)
            outputs.append(self.decoder(slot_h))
        
        # Sum slots (key: additive composition)
        out = sum(outputs)
        
        return out, slots.mean(1)

# ============================================================
# MAIN
# ============================================================

def generate_data(n_seqs=2000, seq_len=10, size=32):
    env = TwoBallsHard(size)
    sequences = []
    for _ in range(n_seqs):
        env.reset()
        seq = [env.render()]
        for _ in range(seq_len-1):
            env.step()
            seq.append(env.render())
        sequences.append(np.array(seq))
    return np.array(sequences)

print("="*60)
print("OBJECT EMERGENCE - V2 SLOT MODEL")
print("="*60)

# Generate data
print("\n1. Generating data...")
train_seqs = generate_data(2000, 10)
test_seqs = generate_data(500, 10)

train_data = torch.FloatTensor(train_seqs).permute(0,1,4,2,3) / 1.0
test_data = torch.FloatTensor(test_seqs).permute(0,1,4,2,3) / 1.0
print(f"Train: {train_data.shape}, Test: {test_data.shape}")

# Train models
models = {
    'Baseline': Baseline(16),
    'SlotV2': SlotModelV2(2),
}

results = {}

for name, model in models.items():
    print(f"\n2. Training {name}...")
    
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(8):
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
        if epoch < 5 or epoch == 7:
            print(f"   Epoch {epoch+1}: loss={np.mean(losses):.4f}")
    
    model.eval()
    with torch.no_grad():
        pred, z = model(test_data[:, 0])
        mse = F.mse_loss(pred, test_data[:, 1]).item()
    
    results[name] = {'mse': mse, 'z_std': z.std().item()}
    print(f"   Test MSE: {mse:.4f}, z_std: {z.std():.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, r in results.items():
    print(f"{name}: MSE={r['mse']:.4f}, z_std={r['z_std']:.4f}")
