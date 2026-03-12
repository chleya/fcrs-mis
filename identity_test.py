"""
Identity Persistence Test - Key Experiment C
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

class TwoIdenticalBalls:
    def __init__(self, size=32):
        self.size = size
        self.radius = 3
        
    def reset(self, crossing=True):
        if crossing:
            # Guaranteed crossing
            self.ball1 = np.array([5.0, 16.0], dtype=np.float32)
            self.ball2 = np.array([27.0, 16.0], dtype=np.float32)
            self.vel1 = np.array([2.0, random.uniform(-0.3, 0.3)], dtype=np.float32)
            self.vel2 = np.array([-2.0, random.uniform(-0.3, 0.3)], dtype=np.float32)
        else:
            # Random non-crossing
            self.ball1 = np.array([random.randint(5, 15), random.randint(5, 27)], dtype=np.float32)
            self.ball2 = np.array([random.randint(17, 27), random.randint(5, 27)], dtype=np.float32)
            self.vel1 = np.random.randn(2).astype(np.float32) * 2
            self.vel2 = np.random.randn(2).astype(np.float32) * 2
            
    def step(self):
        self.ball1 += self.vel1 * 0.5
        self.ball2 += self.vel2 * 0.5
        
        for ball, vel in [(self.ball1, self.vel1), (self.ball2, self.vel2)]:
            for i in range(2):
                if ball[i] < self.radius: ball[i], vel[i] = self.radius, abs(vel[i])
                elif ball[i] > self.size - self.radius: ball[i], vel[i] = self.size - self.radius, -abs(vel[i])
                    
        dist = np.linalg.norm(self.ball1 - self.ball2)
        if dist < self.radius * 2 and dist > 0:
            normal = (self.ball1 - self.ball2) / dist
            impulse = np.dot(self.vel1 - self.vel2, normal)
            self.vel1 -= impulse * normal
            self.vel2 += impulse * normal
            
    def render(self):
        img = np.ones((self.size, self.size, 3), dtype=np.float32) * 0.1
        for ball in [self.ball1, self.ball2]:
            for dx in range(-self.radius, self.radius+1):
                for dy in range(-self.radius, self.radius+1):
                    if dx*dx + dy*dy <= self.radius*self.radius:
                        x, y = int(ball[0]+dx), int(ball[1]+dy)
                        if 0 <= x < self.size and 0 <= y < self.size:
                            img[y, x] = [1.0, 1.0, 1.0]
        return img

def generate_data(n_seqs=15000, seq_len=15, size=32, crossing_ratio=0.5):
    env = TwoIdenticalBalls(size)
    images, targets, crossing_flags = [], [], []
    
    for i in range(n_seqs):
        crossing = (i < n_seqs * crossing_ratio)
        env.reset(crossing)
        
        for step in range(seq_len):
            img = env.render()
            images.append(img.copy())
            
            # Save ball1 future position
            future_ball = env.ball1.copy()
            for _ in range(min(5, seq_len - step - 1)):
                env.step()
            targets.append(future_ball / size)
            
            # Check if crossed
            crossed = False
            if step > 0:
                prev_dist = abs(env.ball1[0] - env.ball2[0])
                if prev_dist < 2:
                    crossed = True
            crossing_flags.append(1 if crossed else 0)
            
            env.step()
    
    return np.array(images), np.array(targets), np.array(crossing_flags)

class Baseline(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Linear(128*4*4, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 2)
        )
        
    def forward(self, x):
        h = self.encoder(x).reshape(x.size(0), -1)
        z = self.fc(h)
        return self.predictor(z), z

class SlotModel(nn.Module):
    def __init__(self, n_slots=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)
        self.mlp = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.predictor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
        
    def forward(self, x):
        h = self.encoder(x).flatten(2).transpose(1, 2)
        slots = torch.randn(x.size(0), 2, 128, device=x.device)
        for _ in range(3):
            q = self.query(slots)
            k = self.key(h).mean(1, keepdim=True)
            v = self.value(h).mean(1, keepdim=True)
            attn = torch.softmax(q @ k.transpose(-2,-1) / 8, -1)
            slots = attn @ v + slots + self.mlp(slots)
        return self.predictor(slots[:, 0, :]), slots.mean(1)

print("="*60)
print("IDENTITY PERSISTENCE TEST")
print("="*60)

print("\n1. Generating data...")
images, targets, crossing = generate_data(15000, 15, 32, 0.5)
idx = np.random.permutation(len(images))
images, targets, crossing = images[idx], targets[idx], crossing[idx]

split = int(0.8 * len(images))
train_img = torch.FloatTensor(images[:split]).permute(0, 3, 1, 2) / 1.0
test_img = torch.FloatTensor(images[split:]).permute(0, 3, 1, 2) / 1.0
train_tgt = torch.FloatTensor(targets[:split])
test_tgt = torch.FloatTensor(targets[split:])
test_cross = crossing[split:]

print(f"Train: {len(train_img)}, Test: {len(test_img)}")
print(f"Crossing ratio: {test_cross.mean():.2%}")

print("\n2. Training models...")

models = {'Baseline': Baseline(16), 'Slot': SlotModel(2)}
results = {}

for name, model in models.items():
    print(f"\n{name}:")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(8):
        idx = torch.randperm(len(train_img))
        losses = []
        for i in range(0, len(idx), 32):
            pred, _ = model(train_img[idx[i:i+32]])
            loss = F.mse_loss(pred, train_tgt[idx[i:i+32]])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if epoch < 4 or epoch == 7:
            print(f"  Epoch {epoch+1}: {np.mean(losses):.4f}")
    
    model.eval()
    with torch.no_grad():
        pred, _ = model(test_img)
        mse = F.mse_loss(pred, test_tgt).item()
        mse_cross = F.mse_loss(pred[test_cross==1], test_tgt[test_cross==1]).item()
        mse_norm = F.mse_loss(pred[test_cross==0], test_tgt[test_cross==0]).item()
        dist = (pred - test_tgt).norm(dim=1).mean().item()
        dist_c = (pred[test_cross==1] - test_tgt[test_cross==1]).norm(dim=1).mean().item()
    
    results[name] = {'mse': mse, 'mse_c': mse_cross, 'mse_n': mse_norm, 'dist': dist, 'dist_c': dist_c}
    print(f"  MSE: {mse:.4f}, MSE(cross): {mse_cross:.4f}, Dist: {dist:.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, r in results.items():
    print(f"{name}: MSE={r['mse']:.4f}, MSE(cross)={r['mse_c']:.4f}, Dist={r['dist']:.4f}")
