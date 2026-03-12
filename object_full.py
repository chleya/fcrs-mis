"""
Object Emergence - Full Comparison
Baseline vs Capacity vs Slot
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

class SlotAttention(nn.Module):
    def __init__(self, in_dim=128, n_slots=2, slot_dim=32):
        super().__init__()
        self.n_slots = n_slots
        self.norm = nn.LayerNorm(in_dim)
        self.query = nn.Linear(slot_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, in_dim))
        
    def forward(self, x, n_iter=3):
        B, D = x.shape[0], x.shape[-1]
        slots = torch.randn(B, self.n_slots, D, device=x.device)
        for _ in range(n_iter):
            q = self.query(slots)
            k = self.key(x).mean(1)
            v = self.value(x).mean(1)
            attn = torch.softmax(q @ k.transpose(-2,-1) / D**0.5, -1)
            slots = attn @ v + slots
            slots = slots + self.mlp(slots + nn.functional.layer_norm(slots, (D,)))
        return slots

class SlotModel(nn.Module):
    def __init__(self, n_slots=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        self.slot_attn = SlotAttention(128, n_slots, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        
    def forward(self, x):
        h = self.encoder(x)  # (B, 128, 4, 4)
        h_flat = h.flatten(2).transpose(1, 2)  # (B, 16, 128)
        slots = self.slot_attn(h_flat)  # (B, n_slots, 128)
        # Decode mean slot
        slot_mean = slots.mean(1).unsqueeze(-1).unsqueeze(-1)
        slot_mean = slot_mean.expand(-1, -1, 4, 4)
        out = self.decoder(slot_mean)
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
print("OBJECT EMERGENCE - FULL COMPARISON")
print("="*60)

# Generate data
print("\n1. Generating data...")
train_seqs = generate_data(2000, 10)
test_seqs = generate_data(500, 10)

train_data = torch.FloatTensor(train_seqs).permute(0,1,4,2,3) / 1.0
test_data = torch.FloatTensor(test_seqs).permute(0,1,4,2,3) / 1.0
print(f"Train: {train_data.shape}, Test: {test_data.shape}")

# Models
models = {
    'Baseline (dim=16)': Baseline(16),
    'Capacity (dim=4)': Baseline(4),
    'Slot (2 slots)': SlotModel(2),
}

results = {}

for name, model in models.items():
    print(f"\n2. Training {name}...")
    
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
