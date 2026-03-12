"""CNN vs MLP test for image/perceptual structure"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)

# Generate position-velocity data
def gen_data(n=500):
    xs, vs = [], []
    for _ in range(n):
        x = np.random.uniform(-1, 1)
        v = np.random.uniform(-0.2, 0.2)
        for _ in range(10):
            s = np.zeros(8)
            px = int((x+1)/2*7)
            if 0<=px<8: s[px] = 1.0
            xs.append(s)
            vs.append(v)
            x = np.clip(x + v, -1, 1)
    return np.array(xs, dtype=np.float32), np.array(vs, dtype=np.float32)

X, V = gen_data(500)
Xt = torch.FloatTensor(X)
print('Data:', X.shape)

# MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
    def forward(self, x):
        return self.net(x)

# CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=1)
        self.fc1 = nn.Linear(8, 8)
    def forward(self, x):
        x = x.unsqueeze(1)
        f = torch.relu(self.conv1(x))
        f = f.mean(-1)
        return torch.relu(self.fc1(f))

m1 = MLP()
m2 = CNN()

o1 = torch.optim.Adam(m1.parameters(), lr=0.01)
for _ in range(10):
    for i in range(0, len(Xt), 32):
        loss = F.mse_loss(m1(Xt[i:i+32]), Xt[i:i+32])
        loss.backward(); o1.step(); o1.zero_grad()

o2 = torch.optim.Adam(m2.parameters(), lr=0.01)
for _ in range(10):
    for i in range(0, len(Xt), 32):
        loss = F.mse_loss(m2(Xt[i:i+32]), Xt[i:i+32])
        loss.backward(); o2.step(); o2.zero_grad()

z1 = m1.net[0](Xt).detach()
z2 = m2.conv1(Xt.unsqueeze(1)).mean(-1).detach()

c1 = np.corrcoef(z1[:,0].numpy(), V)[0,1]
c2 = np.corrcoef(z2[:,0].numpy(), V)[0,1]

print()
print('='*50)
print('CNN vs MLP Variable Emergence')
print('='*50)
print(f'MLP: Corr = {c1:+.3f}')
print(f'CNN: Corr = {c2:+.3f}')
