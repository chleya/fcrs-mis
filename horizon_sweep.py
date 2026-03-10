#!/usr/bin/env python3
"""
Horizon Sweep: Test if longer prediction horizon forces velocity encoding
T ∈ {1, 5, 10, 20, 50}
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

np.random.seed(42)

class Ball:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * 10 + 3
        self.vel = (np.random.rand(2) - 0.5) * 0.4
    
    def step(self):
        self.vel += (np.random.rand(2) - 0.5) * 0.1
        self.vel = np.clip(self.vel, -0.8, 0.8)
        self.pos += self.vel
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)


def generate_data(T):
    """Generate data for prediction horizon T"""
    ball = Ball()
    ball.reset()
    
    # Generate trajectory
    traj = [ball.pos.copy()]
    for _ in range(T + 5):
        ball.step()
        traj.append(ball.pos.copy())
    traj = np.array(traj)
    
    # Input: first 3 frames
    input_seq = traj[:3].flatten()
    # Target: position at T steps
    target = traj[3 + T]
    # Velocity (for analysis)
    velocity = (traj[3+T] - traj[2]) / T
    
    return input_seq, target, velocity


class Model:
    def __init__(self, n_hidden=64, lam=0.01, lr=0.01):
        self.lam = lam
        self.lr = lr
        self.W = np.random.randn(n_hidden, 6) * 0.1
        self.h = np.zeros(n_hidden)
    
    def forward(self, x):
        self.h = np.tanh(self.W @ x)
        return (self.W.T @ self.h)[:2]
    
    def update(self, x, y):
        p = self.forward(x)
        e = y - p
        mse = np.mean(e**2)
        delta = np.mean(e) * np.mean(self.h) - self.lam * np.sign(self.W)
        self.W += self.lr * delta
        self.W[np.abs(self.W) < 1e-4] = 0
        return mse


def train_and_evaluate(T, lam=0.01, steps=3000):
    """Train model for horizon T and evaluate"""
    np.random.seed(42)
    model = Model(n_hidden=64, lam=lam)
    
    # Train
    for _ in range(steps):
        x, y, v = generate_data(T)
        model.update(x, y)
    
    # Evaluate
    h_list, v_list, pos_list = [], [], []
    
    for _ in range(500):
        x, y, v = generate_data(T)
        model.forward(x)
        h_list.append(model.h.copy())
        v_list.append(v.copy())
        pos_list.append(y.copy())
    
    H = np.array(h_list)
    V = np.array(v_list)
    P = np.array(pos_list)
    
    # PCA
    pca = PCA(n_components=min(10, H.shape[1]))
    Hp = pca.fit_transform(H)
    
    # Silhouette
    sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
    
    # ARI with different variables
    v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
    v_bins = np.digitize(v_mag, np.percentile(v_mag, [33, 66]))
    
    p_mag = np.sqrt(P[:,0]**2 + P[:,1]**2)
    p_bins = np.digitize(p_mag, np.percentile(p_mag, [33, 66]))
    
    hidden_clusters = KMeans(3, n_init=10).fit_predict(Hp)
    
    ari_velocity = adjusted_rand_score(hidden_clusters, v_bins)
    ari_position = adjusted_rand_score(hidden_clusters, p_bins)
    
    # MI
    h_mag = np.sqrt((H**2).sum(axis=1))
    mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1]) if len(v_mag)>0 else 0
    mi_v = 0 if np.isnan(mi_v) else mi_v
    
    mi_p = abs(np.corrcoef(p_mag, h_mag)[0,1]) if len(p_mag)>0 else 0
    mi_p = 0 if np.isnan(mi_p) else mi_p
    
    # MSE
    errors = []
    for _ in range(100):
        x, y, v = generate_data(T)
        model.forward(x)
        p = (model.W.T @ model.h)[:2]
        errors.append(np.mean((p - y)**2))
    
    return {
        "T": T,
        "sil": sil,
        "ari_v": ari_velocity,
        "ari_p": ari_position,
        "mi_v": mi_v,
        "mi_p": mi_p,
        "mse": np.mean(errors)
    }


# Main experiment
Ts = [1, 5, 10, 20, 50]
lam = 0.01  # Critical region

print("="*60)
print(f"Horizon Sweep Experiment (λ = {lam})")
print("="*60)

results = []
for T in Ts:
    print(f"\nT = {T}...")
    r = train_and_evaluate(T, lam=lam)
    results.append(r)
    print(f"  Sil={r['sil']:.3f}, ARI(v)={r['ari_v']:.3f}, ARI(p)={r['ari_p']:.3f}")
    print(f"  MI(v)={r['mi_v']:.3f}, MI(p)={r['mi_p']:.3f}, MSE={r['mse']:.3f}")

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"{'T':>4} | {'Sil':>6} | {'ARI(v)':>7} | {'ARI(p)':>7} | {'MI(v)':>6} | {'MI(p)':>6}")
print("-"*60)
for r in results:
    print(f"{r['T']:>4} | {r['sil']:>6.3f} | {r['ari_v']:>7.3f} | {r['ari_p']:>7.3f} | {r['mi_v']:>6.3f} | {r['mi_p']:>6.3f}")

# Trend analysis
print("\n" + "="*60)
print("TREND ANALYSIS:")
print("="*60)
print(f"ARI(v) trend: {results[0]['ari_v']:.3f} → {results[-1]['ari_v']:.3f}")
print(f"ARI(p) trend: {results[0]['ari_p']:.3f} → {results[-1]['ari_p']:.3f}")
print(f"MI(v) trend: {results[0]['mi_v']:.3f} → {results[-1]['mi_v']:.3f}")

if results[-1]['ari_v'] > results[0]['ari_v']:
    print("\n>>> Velocity encoding INCREASES with horizon! <<<")
else:
    print("\n>>> No velocity encoding trend detected <<<")
