"""Simple L2 v2 test"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Simple data
X = np.random.randn(1000, 4)
A = np.random.randint(0, 2, 1000)
Xt = torch.FloatTensor(X)
At = torch.FloatTensor(A).float().unsqueeze(-1)

# Simple model
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(4, 4)
        self.dyn = nn.Linear(5, 4)
        self.dec = nn.Linear(4, 4)
    def forward(self, o, a):
        z = self.enc(o)
        zn = z + self.dyn(torch.cat([z, a], dim=-1))
        return self.dec(zn), z

m = M()
o = torch.optim.Adam(m.parameters(), lr=0.01)
for _ in range(5):
    for i in range(0, 1000, 32):
        p, z = m(Xt[i:i+32], At[i:i+32])
        loss = F.mse_loss(p, Xt[i:i+32])
        loss.backward(); o.step(); o.zero_grad()

print('OK')
