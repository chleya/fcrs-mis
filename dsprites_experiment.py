"""
dSprites Experiment - Variable Emergence in High-Dimensional Image Space

Uses simplified dSprites-like data:
- Factors: shape (3), scale (6), orientation (40), x (32), y (32)
- Observations: 2D binary images

Run: python dsprites_experiment.py
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

print('='*60)
print('dSprites-like VARIABLE EMERGENCE TEST')
print('='*60)

# Generate simplified dSprites data
def generate_dsprites(n_samples=5000, img_size=16):
    """
    Generate dSprites-like images with known latent factors
    Factors: position_x, position_y, scale, orientation
    """
    data = []
    
    for _ in range(n_samples):
        # Random latent factors
        x = np.random.uniform(0.2, 0.8)  # position x
        y = np.random.uniform(0.2, 0.8)  # position y
        scale = np.random.uniform(0.3, 0.7)  # scale
        orientation = np.random.uniform(0, 2*np.pi)  # orientation
        
        # Create image
        img = np.zeros((img_size, img_size))
        
        # Draw ellipse at (x, y) with given scale and orientation
        cx, cy = int(x * img_size), int(y * img_size)
        
        for i in range(img_size):
            for j in range(img_size):
                # Ellipse equation
                dx = (i - cy) / (scale * img_size / 2)
                dy = (j - cx) / (scale * img_size / 2)
                dist = dx**2 + dy**2
                if dist <= 1:
                    img[i, j] = 1.0
        
        # Add slight noise
        img = img + np.random.randn(img_size, img_size) * 0.05
        img = np.clip(img, 0, 1)
        
        # Next frame: small movement
        x_next = np.clip(x + np.random.uniform(-0.05, 0.05), 0.1, 0.9)
        y_next = np.clip(y + np.random.uniform(-0.05, 0.05), 0.1, 0.9)
        
        img_next = np.zeros((img_size, img_size))
        cx_next, cy_next = int(x_next * img_size), int(y_next * img_size)
        
        for i in range(img_size):
            for j in range(img_size):
                dx = (i - cy_next) / (scale * img_size / 2)
                dy = (j - cx_next) / (scale * img_size / 2)
                dist = dx**2 + dy**2
                if dist <= 1:
                    img_next[i, j] = 1.0
        
        img_next = img_next + np.random.randn(img_size, img_size) * 0.05
        img_next = np.clip(img_next, 0, 1)
        
        data.append({
            'img': img.flatten(),
            'img_next': img_next.flatten(),
            'x': x,
            'y': y,
            'x_next': x_next,
            'y_next': y_next,
            'scale': scale,
            'orientation': orientation
        })
    
    return data

# Generate data
print('\nGenerating dSprites-like data...')
data = generate_dsprites(5000, img_size=16)
print(f'Generated {len(data)} samples')

# Prepare tensors
images = np.array([d['img'] for d in data])
images_next = np.array([d['img_next'] for d in data])
positions_x = np.array([d['x'] for d in data])
positions_y = np.array([d['y'] for d in data])
velocities_x = np.array([d['x_next'] - d['x'] for d in data])

X_t = torch.FloatTensor(images)
X_next_t = torch.FloatTensor(images_next)

print(f'Image shape: {images.shape}')

# Models
class Baseline(nn.Module):
    """Baseline: MLP on flattened pixels"""
    def __init__(self, obs_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim))
    
    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))

class CausalMLP(nn.Module):
    """Causal: encoder-dynamics-decoder with MLP"""
    def __init__(self, obs_dim=256, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim))
        
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim))
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim))
    
    def forward(self, obs, action):
        z = self.encoder(obs)
        z_next = z + self.dynamics(torch.cat([z, action], dim=-1))
        return self.decoder(z_next), z

# CNN-based models
class CausalCNN(nn.Module):
    """Causal with CNN encoder"""
    def __init__(self, img_size=16, latent_dim=8):
        super().__init__()
        
        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten())
        
        # Calculate feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            feat_dim = self.encoder(dummy).shape[1]
        
        self.latent = nn.Linear(feat_dim, latent_dim)
        
        # Dynamics
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim))
        
        # Deconv
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, padding=1))
    
    def forward(self, obs, action):
        # obs: (batch, 256)
        x = obs.view(-1, 1, 16, 16)
        feat = self.encoder(x)
        z = torch.relu(self.latent(feat))
        
        z_next = z + self.dynamics(torch.cat([z, action], dim=-1))
        
        decoded = self.decoder(z_next)
        decoded = decoded.view(-1, 32, 8, 8)
        recon = self.deconv(decoded)
        
        return recon.view(-1, 256), z

# Train function
def train_and_evaluate(model, model_type, epochs=10):
    if model_type == 'cnn':
        # CNN needs special handling
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            idx = np.random.permutation(len(X_t))
            for i in range(0, len(idx), 64):
                batch_idx = idx[i:i+64]
                obs = X_t[batch_idx]
                obs_next = X_next_t[batch_idx]
                action = torch.zeros(len(obs), 1)  # No action for now
                
                pred, z = model(obs, action)
                loss = F.mse_loss(pred, obs)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            action = torch.zeros(len(X_t), 1)
            _, z = model(X_t, action)
        
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        action_t = torch.zeros(len(X_t), 1)
        
        for epoch in range(epochs):
            idx = np.random.permutation(len(X_t))
            for i in range(0, len(idx), 64):
                batch_idx = idx[i:i+64]
                obs = X_t[batch_idx]
                obs_next = X_next_t[batch_idx]
                action = action_t[batch_idx]
                
                if model_type == 'baseline':
                    pred = model(obs, action)
                    z = None
                else:
                    pred, z = model(obs, action)
                
                loss = F.mse_loss(pred, obs_next)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            if model_type == 'baseline':
                z = model.net[:3](torch.cat([X_t, action_t], dim=-1))
            else:
                _, z = model(X_t, action_t)
    
    z_np = z[:, 0].detach().numpy()
    
    # Correlation with position x
    corr_x = np.corrcoef(z_np, positions_x)[0, 1]
    corr_y = np.corrcoef(z_np, positions_y)[0, 1]
    corr_vx = np.corrcoef(z_np, velocities_x)[0, 1]
    
    return {
        'corr_x': abs(corr_x) if not np.isnan(corr_x) else 0,
        'corr_y': abs(corr_y) if not np.isnan(corr_y) else 0,
        'corr_vx': abs(corr_vx) if not np.isnan(corr_vx) else 0,
        'mean_corr': np.mean([abs(corr_x), abs(corr_y), abs(corr_vx)])
    }

# Run experiments
print('\n' + '='*60)
print('RESULTS')
print('='*60)
print(f"{'Model':<15} | {'|Corr(x)|':>10} | {'|Corr(vx)|':>10} | {'Mean':>10}")
print('-'*60)

# 1. Baseline (MLP)
print('Training Baseline (MLP)...')
baseline = Baseline(256)
r1 = train_and_evaluate(baseline, 'baseline')
print(f"{'Baseline MLP':<15} | {r1['corr_x']:>10.3f} | {r1['corr_vx']:>10.3f} | {r1['mean_corr']:>10.3f}")

# 2. Causal (MLP)
print('Training Causal (MLP)...')
causal_mlp = CausalMLP(256, 8)
r2 = train_and_evaluate(causal_mlp, 'causal_mlp')
print(f"{'Causal MLP':<15} | {r2['corr_x']:>10.3f} | {r2['corr_vx']:>10.3f} | {r2['mean_corr']:>10.3f}")

# 3. Causal (CNN)
print('Training Causal (CNN)...')
causal_cnn = CausalCNN(16, 8)
r3 = train_and_evaluate(causal_cnn, 'cnn')
print(f"{'Causal CNN':<15} | {r3['corr_x']:>10.3f} | {r3['corr_vx']:>10.3f} | {r3['mean_corr']:>10.3f}")

print('='*60)

# Analysis
print('\nANALYSIS:')
print('-'*40)
if r3['mean_corr'] > r2['mean_corr']:
    print('CNN encoder improves variable emergence')
elif r3['mean_corr'] > r1['mean_corr']:
    print('Causal structure improves variable emergence')
else:
    print('Current structure cannot handle high-dimensional images')

print(f'\nNote: This is a simplified dSprites test.')
print(f'For full dSprites (64x64), would need more training/complex models.')
