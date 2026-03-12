"""
L2 Mechanism v1 - CORRECT METRIC
Restart from scratch with correct evaluation

Correct metric:
- Mechanism = intervention should REDUCE prediction error
- Not variance change, but: does latent help predict intervention effects?

Run: python l2_v1_correct.py
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

def step(s, a):
    x, xd, t, td = s
    f = 10.0 if a==1 else -10.0
    ct, st = np.cos(t), np.sin(t)
    tmp = f/1.1 + 0.05*td**2*st
    ta = (9.8*st - ct*tmp)/(0.5*(4/3-0.1*ct**2/1.1))
    return np.array([x+0.02*xd, xd+0.02*(tmp-0.05*ta*ct/1.1), t+0.02*td, td+0.02*ta])

# Data: normal vs intervention
print('Generating data...')
data = []

# Normal transitions
for _ in range(2000):
    s = np.random.uniform(-0.05, 0.05, 4)
    for _ in range(20):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        data.append({'s': s.copy(), 'a': a, 'sn': sn.copy(), 'type': 'normal'})
        s = sn

# Intervention: fix theta = 0
for _ in range(500):
    s = np.random.uniform(-0.05, 0.05, 4)
    s[2] = 0.0  # theta = 0
    for _ in range(10):
        a = np.random.randint(0, 2)
        sn = step(s, a)
        sn[2] = 0.0  # keep theta fixed
        data.append({'s': s.copy(), 'a': a, 'sn': sn.copy(), 'type': 'theta_fix'})
        s = sn

np.random.shuffle(data)

S = np.array([d['s'] for d in data])
A = np.array([d['a'] for d in data])
SN = np.array([d['sn'] for d in data])
T = np.array([d['type'] for d in data])

St = torch.FloatTensor(S)
At = torch.FloatTensor(A).float()
SNt = torch.FloatTensor(SN)

# Models
class Baseline(nn.Module):
    """No causal structure"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, 4))
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1))

class Causal(nn.Module):
    """Encoder-dynamics-decoder"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dyn = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dec = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 4))
    def forward(self, s, a):
        z = self.enc(s)
        zn = z + self.dyn(torch.cat([z, a], -1))
        return self.dec(zn), z

# Train and evaluate
def train_eval(model, model_type):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for _ in range(10):
        idx = np.random.permutation(len(St))
        for i in range(0, len(idx), 32):
            if model_type == 'baseline':
                pred = model(St[idx[i:i+32]], At[idx[i:i+32]].unsqueeze(-1))
                loss = F.mse_loss(pred, SNt[idx[i:i+32]])
            else:
                pred, _ = model(St[idx[i:i+32]], At[idx[i:i+32]].unsqueeze(-1))
                loss = F.mse_loss(pred, SNt[idx[i:i+32]])
            
            loss.backward()
            opt.step()
            opt.zero_grad()
    
    # Evaluate on normal vs intervention
    model.eval()
    with torch.no_grad():
        if model_type == 'baseline':
            pred = model(St, At.unsqueeze(-1))
        else:
            pred, _ = model(St, At.unsqueeze(-1))
    
    pred_np = pred.numpy()
    
    # Prediction error on normal vs intervention
    normal_idx = T == 'normal'
    int_idx = T == 'theta_fix'
    
    mse_normal = np.mean((pred_np[normal_idx] - SN[normal_idx])**2)
    mse_int = np.mean((pred_np[int_idx] - SN[int_idx])**2)
    
    # Latent correlation
    if model_type == 'causal':
        _, z = model(St, At.unsqueeze(-1))
        z_np = z.detach().numpy()
        
        # Correlation with true states
        corr_vel = np.corrcoef(z_np[:, 0], S[:, 1])[0, 1]
        corr_theta = np.corrcoef(z_np[:, 1], S[:, 2])[0, 1]
        
        return mse_normal, mse_int, corr_vel, corr_theta
    
    return mse_normal, mse_int, None, None

print('\n' + '='*60)
print('L2 V1 - CORRECT EVALUATION')
print('='*60)

# Baseline
print('\nBaseline:')
mse_n, mse_i, _, _ = train_eval(Baseline(), 'baseline')
print(f'  Normal MSE: {mse_n:.4f}')
print(f'  Intervention MSE: {mse_i:.4f}')
print(f'  Ratio (int/normal): {mse_i/mse_n:.2f}')

# Causal
print('\nCausal:')
mse_n, mse_i, corr_v, corr_t = train_eval(Causal(), 'causal')
print(f'  Normal MSE: {mse_n:.4f}')
print(f'  Intervention MSE: {mse_i:.4f}')
print(f'  Ratio (int/normal): {mse_i/mse_n:.2f}')
print(f'  Corr(z, velocity): {corr_v:.3f}')
print(f'  Corr(z, theta): {corr_t:.3f}')

print('\n' + '='*60)
print('CONCLUSION')
print('='*60)
print('\nKey metric: Lower intervention MSE = Better mechanism')
print('If Causal has lower MSE on intervention, mechanism helps!')
