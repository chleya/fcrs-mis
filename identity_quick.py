"""Identity Test - Simplified"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
random.seed(42); np.random.seed(42); torch.manual_seed(42)

class Env:
    def __init__(self):
        self.size, self.r = 32, 3
    def init(self, cross=True):
        if cross:
            self.b1, self.b2 = np.array([5., 16.], dtype=np.float32), np.array([27., 16.], dtype=np.float32)
            self.v1, self.v2 = np.array([2., 0.], dtype=np.float32), np.array([-2., 0.], dtype=np.float32)
        else:
            self.b1, self.b2 = np.array([10., 10.], dtype=np.float32), np.array([22., 22.], dtype=np.float32)
            self.v1 = np.random.randn(2).astype(np.float32) * 2
            self.v2 = np.random.randn(2).astype(np.float32) * 2
    def step(self):
        self.b1, self.b2 = self.b1 + self.v1 * 0.5, self.b2 + self.v2 * 0.5
        for b, v in [(self.b1, self.v1), (self.b2, self.v2)]:
            for i in range(2):
                if b[i] < self.r: b[i], v[i] = self.r, abs(v[i])
                elif b[i] > self.size - self.r: b[i], v[i] = self.size - self.r, -abs(v[i])
        d = np.linalg.norm(self.b1 - self.b2)
        if d < self.r * 2 and d > 0:
            n = (self.b1 - self.b2) / d
            imp = np.dot(self.v1 - self.v2, n)
            self.v1, self.v2 = self.v1 - imp * n, self.v2 + imp * n
    def render(self):
        img = np.ones((self.size, self.size, 3), dtype=np.float32) * 0.1
        for b in [self.b1, self.b2]:
            for dx in range(-self.r, self.r + 1):
                for dy in range(-self.r, self.r + 1):
                    if dx * dx + dy * dy <= self.r * self.r:
                        x, y = int(b[0] + dx), int(b[1] + dy)
                        if 0 <= x < self.size and 0 <= y < self.size:
                            img[y, x] = [1, 1, 1]
        return img

def gen_data(n=6000):
    I, T, C = [], [], []
    e = Env()
    for i in range(n):
        e.init(i < n // 2)
        for _ in range(8):
            I.append(e.render().copy())
            fb = e.b1.copy(); fv = e.v1.copy()
            for _ in range(3):
                e.b1, e.b2 = e.b1 + e.v1 * 0.5, e.b2 + e.v2 * 0.5
                # simple bounce
                for b in [e.b1, e.b2]:
                    for i in range(2):
                        if b[i] < 3: b[i] = 3
                        elif b[i] > 29: b[i] = 29
            T.append(fb / 32)
            C.append(1 if e.b1[0] > e.b2[0] else 0)
            e.step()
    return np.array(I), np.array(T), np.array(C)

print("Generating...")
I, T, C = gen_data(6000)
idx = np.random.permutation(len(I)); I, T, C = I[idx], T[idx], C[idx]
sp = int(0.8 * len(I))
tr, te = I[:sp], I[sp:]; trt, tet = T[:sp], T[sp:]; tc = C[sp:]
tr = torch.FloatTensor(tr).permute(0, 3, 1, 2)
te = torch.FloatTensor(te).permute(0, 3, 1, 2)
trt, tet = torch.FloatTensor(trt), torch.FloatTensor(tet)

class M(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.ReLU(),nn.Conv2d(32,64,4,2,1),nn.ReLU(),nn.Conv2d(64,128,4,2,1),nn.ReLU())
        self.fc = nn.Linear(128*4*4, d)
        self.dec = nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x):
        return self.dec(self.fc(self.enc(x).reshape(x.size(0),-1))), self.fc(self.enc(x).reshape(x.size(0),-1))

print("Training..."); m=M(16); o=torch.optim.Adam(m.parameters(),3e-4)
for e in range(5):
    id=torch.randperm(len(tr))
    for i in range(0,len(id),32):
        l=F.mse_loss(m(tr[id[i:i+32]])[0],trt[id[i:i+32]]); l.backward(); o.step(); o.zero_grad()
    print(f"  {e+1}")
m.eval()
with torch.no_grad():
    p,_=m(te)
    print(f"MSE={F.mse_loss(p,tet):.4f}, MSE(c)={F.mse_loss(p[tc==1],tet[tc==1]):.4f}, MSE(n)={F.mse_loss(p[tc==0],tet[tc==0]):.4f}")
