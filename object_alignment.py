"""
Object Emergence - Test Object Alignment Directly
Track which slot corresponds to which ball
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
        self.color1 = np.array([1.0, 0.0, 0.0])  # Red ball 1
        self.color2 = np.array([0.0, 0.0, 1.0])  # Blue ball 2
        self.radius1 = 3
        self.radius2 = 3
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

# Generate data with tracking
print("="*60)
print("OBJECT ALIGNMENT TEST")
print("="*60)

env = TwoBallsHard(32)
data = []
positions = []

for _ in range(1000):
    env.reset()
    for _ in range(5):
        img = env.render()
        data.append(img.copy())
        positions.append((env.ball1.copy(), env.ball2.copy()))
        env.step()

data = np.array(data)
positions = positions

train_data = torch.FloatTensor(data).permute(0, 3, 1, 2) / 1.0

# Train
print("\n1. Training SlotModel...")
model = SlotModel(2)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(10):
    idx = torch.randperm(len(train_data))
    losses = []
    for i in range(0, len(idx), 32):
        batch = train_data[idx[i:i+32]]
        pred, slots = model(batch, return_slots=True)
        loss = F.mse_loss(pred, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if epoch < 5 or epoch == 9:
        print(f"   Epoch {epoch+1}: loss={np.mean(losses):.4f}")

# Test object alignment
print("\n2. Testing Object Alignment...")
model.eval()

test_data = []
test_positions = []
for _ in range(200):
    env.reset()
    for _ in range(5):
        test_data.append(env.render().copy())
        test_positions.append((env.ball1.copy(), env.ball2.copy()))
        env.step()

test_data = torch.FloatTensor(np.array(test_data)).permute(0, 3, 1, 2) / 1.0
test_positions = test_positions[:200]  # Take first 200

with torch.no_grad():
    _, slots = model(test_data, return_slots=True)
    slots_np = slots.numpy()  # (200, 2, 128)

# Compute correlation between each slot and each ball's position
print("\n3. Correlation Analysis:")
ball1_x = np.array([p[0][0] for p in test_positions])
ball1_y = np.array([p[0][1] for p in test_positions])
ball2_x = np.array([p[1][0] for p in test_positions])
ball2_y = np.array([p[1][1] for p in test_positions])

for slot_idx in range(2):
    slot_repr = slots_np[:, slot_idx, :]  # (200, 128)
    
    # Project to 2D using PCA-like approach (just use first 2 dims)
    slot_2d = slot_repr[:, :2]
    
    corr_x1 = np.corrcoef(slot_2d[:, 0], ball1_x)[0, 1]
    corr_y1 = np.corrcoef(slot_2d[:, 1], ball1_y)[0, 1]
    corr_x2 = np.corrcoef(slot_2d[:, 0], ball2_x)[0, 1]
    corr_y2 = np.corrcoef(slot_2d[:, 1], ball2_y)[0, 1]
    
    print(f"\nSlot {slot_idx}:")
    print(f"  vs Ball1 (x,y): {corr_x1:+.3f}, {corr_y1:+.3f}")
    print(f"  vs Ball2 (x,y): {corr_x2:+.3f}, {corr_y2:+.3f}")
    
    # Check if slot aligns with specific ball
    best_ball1 = max(abs(corr_x1), abs(corr_y1))
    best_ball2 = max(abs(corr_x2), abs(corr_y2))
    print(f"  Best match: Ball{'1' if best_ball1 > best_ball2 else '2'} ({max(best_ball1, best_ball2):.3f})")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("If slots correlate with ball positions -> object alignment!")
print("If slots don't correlate -> no object emergence")
