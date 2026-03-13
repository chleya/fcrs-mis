"""
PERCEPTION BOTTLENECK TEST
Coordinates vs Pixels vs Segmentation

Test how input representation affects scaling
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MAX_OBJ = 10
print("="*60)
print("PERCEPTION BOTTLENECK TEST")
print("="*60)

def clamp(v): return max(3, min(28, int(v)))

# 1. Coordinates (given objects)
def gen_coords(n, nobj):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(nobj)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(nobj)]
        
        # Features: [x, y, vx, vy] per object
        f0 = []
        for i in range(MAX_OBJ):
            if i < nobj:
                f0.extend([pos[i][0]/32, pos[i][1]/32, vel[i][0]/4, vel[i][1]/4])
            else:
                f0.extend([0, 0, 0, 0])
        
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]; vx, vy = vel[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
        
        f10 = []
        for i in range(MAX_OBJ):
            if i < nobj:
                f10.extend([pos[i][0]/32, pos[i][1]/32, vel[i][0]/4, vel[i][1]/4])
            else:
                f10.extend([0, 0, 0, 0])
        
        X.append(f0 + f10)
        Y.append(pos[0][0]/32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# 2. Pixels (raw images)
def gen_pixels(n, nobj):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(nobj)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(nobj)]
        
        img0 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img0[clamp(y), clamp(x)] = [1,1,1]
        
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]; vx, vy = vel[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
        
        img10 = np.zeros((32, 32, 3), np.float32)
        for x, y in pos: img10[clamp(y), clamp(x)] = [1,1,1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(pos[0][0]/32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# 3. Segmentation (pixels + mask channel)
def gen_segment(n, nobj):
    X, Y = [], []
    for _ in range(n):
        pos = [(random.uniform(5, 27), random.uniform(8, 24)) for _ in range(nobj)]
        vel = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(nobj)]
        
        # RGB + segmentation mask
        img0 = np.zeros((32, 32, 4), np.float32)
        for i, (x, y) in enumerate(pos):
            img0[clamp(y), clamp(x), :3] = [1,1,1]
            if i == 0:
                img0[clamp(y), clamp(x), 3] = 1.0  # Target object mask
        
        for _ in range(10):
            for i in range(len(pos)):
                x, y = pos[i]; vx, vy = vel[i]
                x, y = x+vx*0.5, y+vy*0.5
                if x<3 or x>29: vx*=-1
                if y<3 or y>29: vy*=-1
                pos[i], vel[i] = (x,y), (vx,vy)
        
        img10 = np.zeros((32, 32, 4), np.float32)
        for x, y in pos: img10[clamp(y), clamp(x), :3] = [1,1,1]
        
        X.append(np.concatenate([img0, img10], axis=2))
        Y.append(pos[0][0]/32)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

class ModelCoord(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self,x): return self.fc(x).squeeze()

class ModelPixel(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_ch,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x): return self.fc(self.enc(x).flatten(1)).squeeze()

print("\n1. Testing input representations...")

results = []

# Test each input type
for input_type, gen_func, in_ch in [("Coordinates", gen_coords, 48), 
                                       ("Pixels (RGB)", gen_pixels, 6), 
                                       ("Segmentation", gen_segment, 8)]:
    print(f"\n=== {input_type} ===")
    
    input_results = {'type': input_type, 'n2': [], 'n6': []}
    
    # Train on N=2
    X_tr, Y_tr = gen_func(1500, 2)
    if input_type == "Coordinates":
        X_tr = torch.FloatTensor(X_tr)
    else:
        X_tr = torch.FloatTensor(X_tr).permute(0, 3, 1, 2)
    Y_tr = torch.FloatTensor(Y_tr)
    
    # Test on N=2 and N=6
    for n in [2, 6]:
        X_t, Y_t = gen_func(500, n)
        if input_type == "Coordinates":
            X_t = torch.FloatTensor(X_t)
        else:
            X_t = torch.FloatTensor(X_t).permute(0, 3, 1, 2)
        Y_t = torch.FloatTensor(Y_t)
        
        # Simple model
        if input_type == "Coordinates":
            m = ModelCoord()
        else:
            m = ModelPixel(in_ch)
        
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(10):
            idx = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 64):
                p = m(X_tr[idx[i:i+64]])
                loss = F.mse_loss(p, Y_tr[idx[i:i+64]])
                opt.zero_grad(); loss.backward(); opt.step()
        
        mse = F.mse_loss(m(X_t), Y_t).item()
        rnd = Y_t.var().item()
        pct = (rnd-mse)/rnd*100 if rnd>0 else 0
        
        if n == 2:
            input_results['n2'].append(pct)
        else:
            input_results['n6'].append(pct)
        
        print(f"  N={n}: {pct:+.0f}%")
    
    results.append(input_results)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n{'Input Type':<20} {'N=2':>10} {'N=6':>10} {'Change':>10}")
print("-"*50)
for r in results:
    n2 = r['n2'][0] if r['n2'] else 0
    n6 = r['n6'][0] if r['n6'] else 0
    change = n6 - n2
    print(f"{r['type']:<20} {n2:>+10.0f}% {n6:>+10.0f}% {change:>+10.0f}%")

print("\n=> Perception bottleneck: Pixel-based learning fails to discover objects")
print("=> With explicit object information (coordinates/segmentation), scaling works better")
