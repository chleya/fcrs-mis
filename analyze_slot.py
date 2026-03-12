"""
Analysis: Why is Slot so bad?
Let me check what each model is actually doing
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("ANALYSIS: Why Slot is Bad")
print("="*60)

# Simple data
X0 = torch.randn(100, 3, 32, 32)
X5 = torch.randn(100, 3, 32, 32)
X10 = torch.randn(100, 3, 32, 32)
Y = torch.rand(100)

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64*8*8*3, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x0, x5, x10):
        return self.fc(torch.cat([self.enc(x0).flatten(1), self.enc(x5).flatten(1), self.enc(x10).flatten(1)], dim=1)).squeeze()

class SlotBroken(nn.Module):
    """OLD BROKEN VERSION - only uses t0!"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.slot = nn.Parameter(torch.randn(2, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x0, x5, x10):
        # PROBLEM: Only uses x0! Ignores x5 and x10!
        h0 = self.enc(x0).mean(dim=[2, 3]).unsqueeze(1) + self.slot.unsqueeze(0)
        return self.predict(h0[:, 0]).squeeze()

class SlotFixed(nn.Module):
    """FIXED VERSION - uses all frames"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.slot = nn.Parameter(torch.randn(2, 64) * 0.1)
        self.predict = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x0, x5, x10):
        # Process all frames
        h0 = self.enc(x0).mean(dim=[2, 3]).unsqueeze(1) + self.slot.unsqueeze(0)  # (B, 2, 64)
        h5 = self.enc(x5).mean(dim=[2, 3]).unsqueeze(1) + self.slot.unsqueeze(0)
        h10 = self.enc(x10).mean(dim=[2, 3]).unsqueeze(1) + self.slot.unsqueeze(0)
        
        # Use slot 0 for prediction
        p0 = self.predict(h0[:, 0])
        p5 = self.predict(h5[:, 0])
        p10 = self.predict(h10[:, 0])
        
        return ((p0 + p5 + p10) / 3).squeeze()

print("Testing different models with SAME data:")

# Test Baseline
m = Baseline()
p = m(X0, X5, X10)
print(f"\nBaseline uses: x0 + x5 + x10 = {p.shape}")

# Test Broken Slot
m = SlotBroken()
p = m(X0, X5, X10)
print(f"SlotBroken uses: ONLY x0 = {p.shape}")

# Test Fixed Slot
m = SlotFixed()
p = m(X0, X5, X10)
print(f"SlotFixed uses: x0 + x5 + x10 = {p.shape}")

print("\n" + "="*60)
print("ROOT CAUSE FOUND!")
print("="*60)
print("\nSlotBroken code:")
print("  h0 = self.enc(x0)...")
print("  return self.predict(h0[:, 0])  # ONLY uses x0!")
print("\nBut Baseline uses:")
print("  torch.cat([enc(x0), enc(x5), enc(x10)], dim=1)")
print("\n=> Unfair comparison!")
print("=> Slot was only seeing 1/3 of the temporal information!")
