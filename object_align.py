"""
Object Emergence - Test Object Alignment Directly
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

class TwoBallsHard:
    def __init__(self, size=32):
        self.size = size
        
    def reset(self):
        self.color1 = np.array([1.0, 0.0, 0.0])
        self.color2 = np.array([0.0, 0.0, 1.0])
        self.radius1, self.radius2 = 3, 3
        self.ball1 = np.array([np.random.randint(5, 15), np.random.randint(5, 27)], dtype=np.float32)
        self.ball2 = np.array([np.random.randint(17, 27), np.random.randint(5, 27)], dtype=np.float32)
        self.vel1 = np.random.randn(2).astype(np.float32) * 2
        self.vel2 = np.random.randn(2).astype(np.float32) * 2
        self.bg_color = np.array([0.1, 0.1, 0.1])
        
    def step(self):
        self.ball1 += self.vel1 * 0.5
        self.ball2 += self.vel2 * 0.5
        for ball, vel, r in [(self.ball1, self.vel1, self.radius1), (self.ball2, self.vel2, self.radius2)]:
            for i in range(2):
                if ball[i] < r: ball[i], vel[i] = r, abs(vel[i])
                elif ball[i] > self.size-r: ball[i], vel[i] = self.size-r, -abs(vel[i])
        dist = np.linalg.norm(self.ball1 - self.ball2)
        if dist < 6 and dist > 0:
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
        self.slot_dec = nn.Linear(128, 128*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        
    def forward(self, x, return_slots=False):
        h = self.encoder(x)
        h_flat = h.flatten(2).transpose(1, 2)
        slots = torch.randn(x.size(0), 2, 128, device=x.device)
        for _ in range(3):
            q = self.query(slots)
            k = self.key(h_flat).mean(1, keepdim=True)
            v = self.value(h_flat).mean(1, keepdim=True)
            attn = torch.softmax(q @ k.transpose(-2,-1) / 8, -1)
            slots = attn @ v + slots
            slots = slots + self.mlp(slots)
        outputs = []
        for i in range(2):
            slot_h = self.slot_dec(slots[:, i]).reshape(-1, 128, 4, 4)
            outputs.append(self.decoder(slot_h))
        out = sum(outputs)
        if return_slots:
            return out, slots
        return out, slots.mean(1)

# Generate data
print("="*60)
print("OBJECT ALIGNMENT TEST")
print("="*60)

env = TwoBallsHard(32)

# Training data
train_data = []
for _ in range(1000):
    env.reset()
    for _ in range(5):
        train_data.append(env.render())
        env.step()
train_data = torch.FloatTensor(np.array(train_data)).permute(0, 3, 1, 2) / 1.0

# Test data with positions
test_data = []
test_ball1 = []
test_ball2 = []
for _ in range(200):
    env.reset()
    for _ in range(5):
        test_data.append(env.render())
        test_ball1.append(env.ball1.copy())
        test_ball2.append(env.ball2.copy())
        env.step()

test_data = torch.FloatTensor(np.array(test_data)).permute(0, 3, 1, 2) / 1.0
test_ball1 = np.array(test_ball1)
test_ball2 = np.array(test_ball2)

print(f"Train: {train_data.shape}, Test: {test_data.shape}")

# Train
print("\n1. Training SlotModel...")
model = SlotModel(2)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(10):
    idx = torch.randperm(len(train_data))
    losses = []
    for i in range(0, len(idx), 32):
        batch = train_data[idx[i:i+32]]
        pred, _ = model(batch)
        loss = F.mse_loss(pred, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if epoch < 5 or epoch == 9:
        print(f"   Epoch {epoch+1}: loss={np.mean(losses):.4f}")

# Test alignment
print("\n2. Testing Object Alignment...")
model.eval()
with torch.no_grad():
    _, slots = model(test_data, return_slots=True)
    slots_np = slots.numpy()  # (200, 2, 128)

# Correlation
print("\n3. Correlation Analysis:")
ball1_x, ball1_y = test_ball1[:, 0], test_ball1[:, 1]
ball2_x, ball2_y = test_ball2[:, 0], test_ball2[:, 1]

for slot_idx in range(2):
    slot_feat = slots_np[:, slot_idx, :]  # (200, 128)
    
    # Use mean of slot as representation
    slot_mean = slot_feat.mean(axis=1)  # (200,)
    
    # Compute correlation with ball positions
    corr_1x = np.corrcoef(slot_mean, ball1_x)[0,1]
    corr_1y = np.corrcoef(slot_mean, ball1_y)[0,1]
    corr_2x = np.corrcoef(slot_mean, ball2_x)[0,1]
    corr_2y = np.corrcoef(slot_mean, ball2_y)[0,1]
    
    print(f"\nSlot {slot_idx}:")
    print(f"  Corr with Ball1 (x,y): {corr_1x:+.3f}, {corr_1y:+.3f}")
    print(f"  Corr with Ball2 (x,y): {corr_2x:+.3f}, {corr_2y:+.3f}")
    
    best1 = max(abs(corr_1x), abs(corr_1y))
    best2 = max(abs(corr_2x), abs(corr_2y))
    print(f"  Best match: Ball{'1' if best1 > best2 else '2'} ({max(best1,best2):.3f})")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("High correlation (>0.5) = object alignment!")
print("Low correlation = no object emergence")
