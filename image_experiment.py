"""
Image Dataset Test - Variable Emergence with High-Dimensional Data

Test whether causal structure enables variable emergence in image domain.

Simple synthetic dataset:
- Hidden variables: position (x), velocity (v)
- Observation: image (x rendered as 8x8 pixel)
- Task: Predict next frame

Run: python image_experiment.py
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
print('IMAGE DATASET VARIABLE EMERGENCE TEST')
print('='*60)

# ============================================================
# Generate Simple Image Dataset
# ============================================================
def generate_ball_images(num_samples=2000, image_size=8):
    """
    Generate images of a ball at different positions.
    Hidden variables: x position, velocity
    """
    images = []
    positions = []
    velocities = []
    
    for _ in range(num_samples):
        # Random position and velocity
        x = np.random.uniform(-1, 1)
        v = np.random.uniform(-0.2, 0.2)
        
        # Create image (simple Gaussian blob)
        img = np.zeros((image_size, image_size))
        
        # Map position to pixel coordinates
        px = int((x + 1) / 2 * (image_size - 1))
        py = image_size // 2
        
        # Add Gaussian blob
        for i in range(image_size):
            for j in range(image_size):
                dist = np.sqrt((i - py)**2 + (j - px)**2)
                img[i, j] = np.exp(-dist**2 / 2)
        
        # Add noise
        img = img + np.random.randn(image_size, image_size) * 0.1
        img = np.clip(img, 0, 1)
        
        images.append(img.flatten())
        positions.append(x)
        velocities.append(v)
    
    return np.array(images), np.array(positions), np.array(velocities)

# Generate dataset
print('Generating image dataset...')
images, positions, velocities = generate_ball_images(2000)

# Create sequences (current frame -> next frame)
X = images[:-1]  # current frame
V = velocities[:-1]  # current velocity (hidden)
A = np.random.randint(0, 2, len(X))  # random action (0 or 1)
X_next = images[1:]  # next frame

print(f'Dataset: {len(X)} samples')
print(f'Image shape: {images.shape[1]} (8x8=64)')

# Convert to tensors
X_t = torch.FloatTensor(X)
V_t = torch.FloatTensor(V)
A_t = torch.FloatTensor(A).float().unsqueeze(-1)
X_next_t = torch.FloatTensor(X_next)

# ============================================================
# Models
# ============================================================

class Baseline(nn.Module):
    """Baseline: action concatenated with image"""
    def __init__(self, obs_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, obs_dim))
    
    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class Causal(nn.Module):
    """Causal: encoder -> dynamics -> decoder"""
    def __init__(self, obs_dim=64, latent_dim=4):
        super().__init__()
        # Encoder: image -> latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim))
        
        # Dynamics: latent + action -> next latent
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim))
        
        # Decoder: latent -> image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, obs_dim))
    
    def forward(self, obs, action):
        z = self.encoder(obs)
        z_next = z + self.dynamics(torch.cat([z, action], dim=-1))
        return self.decoder(z_next), z


# ============================================================
# Training
# ============================================================

def train_model(model, is_causal, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(len(X_t))
        
        for i in range(0, len(idx), 32):
            batch_idx = idx[i:i+32]
            
            obs = X_t[batch_idx]
            action = A_t[batch_idx]
            target = X_next_t[batch_idx]
            
            if is_causal:
                pred, z = model(obs, action)
            else:
                pred = model(obs, action)
                z = model.net[:3](torch.cat([obs, action], dim=-1))
            
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, is_causal):
    model.eval()
    
    with torch.no_grad():
        if is_causal:
            pred, z = model(X_t, A_t)
        else:
            pred = model(X_t, A_t)
            z = model.net[:3](torch.cat([X_t, A_t], dim=-1))
    
    # Correlation with true velocity
    z_np = z[:, 0].numpy()  # First latent dimension
    
    corr = np.corrcoef(z_np, V)[0, 1]
    corr_abs = abs(corr)
    
    return corr, corr_abs


# ============================================================
# Run Experiment
# ============================================================

print('\nTraining Baseline...')
baseline = Baseline(64)
baseline = train_model(baseline, is_causal=False, epochs=10)
corr_baseline, corr_abs_baseline = evaluate(baseline, is_causal=False)

print('Training Causal...')
causal = Causal(64, 4)
causal = train_model(causal, is_causal=True, epochs=10)
corr_causal, corr_abs_causal = evaluate(causal, is_causal=True)

# ============================================================
# Results
# ============================================================

print('\n' + '='*60)
print('RESULTS')
print('='*60)
print(f"{'Model':<15} | {'Corr(z, v)':>12} | {'|Corr|':>10}")
print('-'*60)
print(f"{'Baseline':<15} | {corr_baseline:>+12.3f} | {corr_abs_baseline:>10.3f}")
print(f"{'Causal':<15} | {corr_causal:>+12.3f} | {corr_abs_causal:>10.3f}")
print('='*60)

print('\nANALYSIS:')
print('-'*60)
if abs(corr_causal) > abs(corr_baseline):
    print(f'Causal improves variable emergence by {abs(corr_causal) - abs(corr_baseline):.3f}')
else:
    print('No improvement - may need more training or different architecture')

print('\n' + '='*60)
print('CONCLUSION')
print('='*60)
print('Testing whether theory generalizes to image domain...')
