"""Identity Test - Simplified"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
random.seed(42); np.random.seed(42); torch.manual_seed(42)

class Env:
    def __init__(self, size=32):
        self.size, self.r = size, 3
    def reset(self, cross=True):
        if cross:
            self.b1, self.b2 = np.array([5., 16.]), np.array([27., 16.])
            self.v1, self.v2 = np.array([2., 0.]), np.array([-2., 0.])
        else:
            self.b1, self.b2 = np.array([10., 10.]), np.array([22., 22.])
            self.v1 = np.random.randn(2).astype(np.float32) * 2
            self.v2 = np.random.randn(2).astype(np.float32) * 2
    def step(self):
        self.b1 = self.b1 + self.v1 * 0.5
        self.b2 = self.b2 + self.v2 * 0.5
        for ball, vel in [(self.b1, self.v1), (self.b2, self.v2)]:
            for i in range(2):
                if ball[i] < self.r:
                    ball[i] = self.r
                    vel[i] = abs(vel[i])
                elif ball[i] > self.size - self.r:
                    ball[i] = self.size - self.r
                    vel[i] = -abs(vel[i])
        dist = np.linalg.norm(self.b1 - self.b2)
        if dist < self.r * 2 and dist > 0:
            normal = (self.b1 - self.b2) / dist
            impulse = np.dot(self.v1 - self.v2, normal)
            self.v1 = self.v1 - impulse * normal
            self.v2 = self.v2 + impulse * normal
    def render(self):
        img = np.ones((self.size, self.size, 3), dtype=np.float32) * 0.1
        for ball in [self.b1, self.b2]:
            for dx in range(-self.r, self.r + 1):
                for dy in range(-self.r, self.r + 1):
                    if dx * dx + dy * dy <= self.r * self.r:
                        x, y = int(ball[0] + dx), int(ball[1] + dy)
                        if 0 <= x < self.size and 0 <= y < self.size:
                            img[y, x] = [1, 1, 1]
        return img

def gen_data(n=8000, seq=10):
    imgs, tgts, cr = [], [], []
    e = Env(32)
    for i in range(n):
        e.reset(i < n // 2)
        for _ in range(seq):
            imgs.append(e.render().copy())
            # Save future position
            f = Env(32)
            f.b1, f.v1 = e.b1.copy(), e.v1.copy()
            for _ in range(3):
                f.step()
            tgts.append(f.b1 / 32)
            # Check crossing
            cr.append(1 if e.b1[0] > e.b2[0] else 0)
            e.step()
    return np.array(imgs), np.array(tgts), np.array(cr)

print("Generating...")
I, T, C = gen_data(8000)
idx = np.random.permutation(len(I))
I, T, C = I[idx], T[idx], C[idx]
tr, te = I[:6400], I[6400:]
trt, tet = T[:6400], T[6400:]
tc = C[6400:]
tr = torch.FloatTensor(tr).permute(0, 3, 1, 2) / 1.0
te = torch.FloatTensor(te).permute(0, 3, 1, 2) / 1.0
trt = torch.FloatTensor(trt)
tet = torch.FloatTensor(tet)

class M(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU()
        )
        self.fc = nn.Linear(128 * 4 * 4, dim)
        self.dec = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        h = self.enc(x).reshape(x.size(0), -1)
        return self.dec(self.fc(h)), self.fc(h)

print("Training Baseline...")
m = M(16)
o = torch.optim.Adam(m.parameters(), 3e-4)
for e in range(6):
    idx = torch.randperm(len(tr))
    for i in range(0, len(idx), 32):
        p, _ = m(tr[idx[i:i + 32]])
        F.mse_loss(p, trt[idx[i:i + 32]]).backward()
        o.step()
        o.zero_grad()
    print(f"  Epoch {e + 1}")

m.eval()
with torch.no_grad():
    p, _ = m(te)
    mse = F.mse_loss(p, tet).item()
    mc = F.mse_loss(p[tc == 1], tet[tc == 1]).item()
    mn = F.mse_loss(p[tc == 0], tet[tc == 0]).item()
print(f"Results: MSE={mse:.4f}, MSE(cross)={mc:.4f}, MSE(normal)={mn:.4f}")
