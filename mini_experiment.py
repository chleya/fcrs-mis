import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

# Simple CartPole Environment
class SimpleCartPole:
    def __init__(self):
        self.gravity = 9.8
        self.length = 0.5
        self.force_mag = 10.0
        self.tau = 0.02
        self.x_threshold = 2.4
        self.theta_threshold = 0.20944
    
    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=(4,))
        return self.state.copy()
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = force / 1.1 + 0.05 * theta_dot**2 * sintheta
        thetaacc = (9.8*sintheta - costheta*temp) / (self.length * (4/3 - 0.1*costheta**2/1.1))
        xacc = temp - 0.05*thetaacc*costheta/1.1
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        done = abs(x) > 2.4 or abs(theta) > 0.20944
        return self.state.copy(), done

# Generate Dataset
def generate_data(num=2000, max_steps=30):
    env = SimpleCartPole()
    obs_transform = np.random.randn(4, 16) * 0.1
    
    data = []
    for _ in range(num):
        v = env.reset()
        for _ in range(max_steps):
            a = np.random.randint(0, 2)
            o = v @ obs_transform + np.random.randn(16) * 0.05
            v_next, done = env.step(a)
            data.append((v.copy(), o.copy(), a))
            v = v_next
            if done: break
    
    # Create samples
    samples = []
    for i in range(len(data) - 3):
        samples.append({
            "v_t": torch.FloatTensor(data[i][0]),
            "o_t": torch.FloatTensor(data[i][1]),
            "a_t": torch.tensor(data[i][2], dtype=torch.float32),
            "v_t1": torch.FloatTensor(data[i+1][0]),
            "o_t1": torch.FloatTensor(data[i+1][1]),
            "v_t2": torch.FloatTensor(data[i+2][0]),
        })
    return samples

print("Generating data...")
data = generate_data(2000)
print(f"Samples: {len(data)}")

class DS(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]

train_loader = DataLoader(DS(data[:1600]), batch_size=32, shuffle=True)
test_loader = DataLoader(DS(data[1600:]), batch_size=32)

# Baseline Model
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16+1, 24),
            nn.ReLU(),
            nn.Linear(24, 4),
            nn.ReLU(),
            nn.Linear(4, 16))
    def forward(self, o, a):
        h = self.net(torch.cat([o, a], dim=-1))
        return h

# Causal Model
class Causal(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 4))
        self.dynamics = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 16))
    def forward(self, o, a):
        z = self.encoder(o)
        z_next = z + self.dynamics(torch.cat([z, a], dim=-1))
        return self.decoder(z_next), z

# Train
def train(model, is_causal, epochs=8):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for ep in range(epochs):
        for batch in train_loader:
            o = batch["o_t"]; a = batch["a_t"].unsqueeze(-1)
            if is_causal:
                pred, z = model(o, a)
                loss = F.mse_loss(pred, batch["o_t1"])
            else:
                pred = model(o, a)
                loss = F.mse_loss(pred, batch["o_t1"])
            opt.zero_grad(); loss.backward(); opt.step()
    return model

# Test
def test(model, is_causal):
    model.eval()
    all_z, all_v = [], []
    with torch.no_grad():
        for batch in test_loader:
            o = batch["o_t"]; a = batch["a_t"].unsqueeze(-1)
            if is_causal:
                pred, z = model(o, a)
            else:
                pred = model(o, a)
                z = model.net[:3](torch.cat([o, a], dim=-1))
            all_z.append(z); all_v.append(batch["v_t"])
    z = torch.cat(all_z, dim=0).numpy()
    v = torch.cat(all_v, dim=0).numpy()
    corr = np.corrcoef(z.mean(axis=1), v[:,0])[0,1]
    return corr

print("\n=== Training Baseline ===")
m1 = Baseline()
m1 = train(m1, False)
corr1 = test(m1, False)
print(f"Baseline Corr(z,v): {corr1:.3f}")

print("\n=== Training Causal ===")
m2 = Causal()
m2 = train(m2, True)
corr2 = test(m2, True)
print(f"Causal Corr(z,v): {corr2:.3f}")

print("\n" + "="*50)
print("RESULT:")
print("="*50)
print(f"Baseline: Corr = {corr1:.3f}")
print(f"Causal:   Corr = {corr2:.3f}")
print()
if corr2 > corr1 * 2:
    print("SUCCESS: Causal architecture improves variable discovery!")
else:
    print("Need more training or different architecture")
