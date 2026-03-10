#!/usr/bin/env python3
"""
Capacity Sweep: Test different latent dimensions
dim ∈ {64, 32, 16, 8, 4, 2}
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


def generate_data(T=20):
    """Generate data for horizon T"""
    ball = Ball()
    ball.reset()
    
    traj = [ball.pos.copy()]
    for _ in range(T + 5):
        ball.step()
        traj.append(ball.pos.copy())
    traj = np.array(traj)
    
    x = traj[:3].flatten()
    y = traj[3 + T]
    v = (traj[3 + T] - traj[2 + T - 1])
    
    return x, y, v


def train_and_evaluate(hidden_dim, T=20, lam=0.01, steps=3000):
    """Train and evaluate with given hidden dimension"""
    np.random.seed(42)
    
    # Initialize model
    W = np.random.randn(hidden_dim, 6) * 0.1
    h = np.zeros(hidden_dim)
    
    # Train
    for _ in range(steps):
        x, y, v = generate_data(T)
        h = np.tanh(W @ x)
        e = y - (W.T @ h)[:2]
        
        # Gradient with compression
        delta = np.mean(e) * np.mean(h) - lam * np.sign(W)
        W += 0.01 * delta
        
        # Clip
        W = np.clip(W, -10, 10)
        W[np.abs(W) < 1e-4] = 0
    
    # Evaluate
    h_list, v_list, pos_list = [], [], []
    
    for _ in range(500):
        x, y, v = generate_data(T)
        h = np.tanh(W @ x)
        
        h_list.append(h.copy())
        v_list.append(v.copy())
        pos_list.append(y.copy())
    
    H = np.array(h_list)
    V = np.array(v_list)
    P = np.array(pos_list)
    
    # Check for collapse
    if len(H) < 10 or np.any(np.isnan(H)):
        return {"dim": hidden_dim, "collapsed": True}

    if len        return {"dim": hidden_dim, "collapsed": True}
    
    # PCA
    try:
        n_comp = min(hidden_dim, 10, H.shape[0] - 1)
        if n_comp < 2:
            return {"dim": hidden_dim, "collapsed": True}
        pca = PCA(n_components=n_comp)
        Hp = pca.fit_transform(H)
        
        sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
    except:
        sil = 0
    
    # ARI
    try:
        v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
        if len(set(np.digitize(v_mag, [np.percentile(v_mag, 33), np.percentile(v_mag, 66)]))) < 2:
            ari_v = 0
        else:
            v_bins = np.digitize(v_mag, [np.percentile(v_mag, 33), np.percentile(v_mag, 66)])
            hidden_clusters = KMeans(3, n_init=10).fit_predict(H)
            ari_v = adjusted_rand_score(hidden_clusters, v_bins)
    except:
        ari_v = 0
    
    # MI
    try:
        h_mag = np.sqrt((H**2).sum(axis=1))
        mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1])
        mi_v = 0 if np.isnan(mi_v) else mi_v
        
        p_mag = np.sqrt(P[:,0]**2 + P[:,1]**2)
        mi_p = abs(np.corrcoef(p_mag, h_mag)[0,1])
        mi_p = 0 if np.isnan(mi_p) else mi_p
    except:
        mi_v = 0
        mi_p = 0
    
    # MSE
    errors = []
    for _ in range(100):
        x, y, v = generate_data(T)
        h = np.tanh(W @ x)
        p = (W.T @ h)[:2]
        errors.append(np.mean((p - y)**2))
    mse = np.mean(errors)
    
    return {
        "dim": hidden_dim,
        "collapsed": False,
        "sil": sil,
        "ari_v": ari_v,
        "mi_v": mi_v,
        "mi_p": mi_p,
        "mse": mse
    }


# Main experiment
dims = [64, 32, 16, 8, 4, 2]
T = 20  # Long horizon
lam = 0.01

print("="*60)
print(f"Capacity Sweep: T={T}, λ={lam}")
print("="*60)

results = []
for dim in dims:
    print(f"\nTesting dim={dim}...")
    r = train_and_evaluate(dim, T=T, lam=lam)
    results.append(r)
    
    if r.get("collapsed"):
        print(f"  COLLAPSED!")
    else:
        print(f"  Sil={r['sil']:.3f}, ARI(v)={r['ari_v']:.3f}")
        print(f"  MI(v)={r['mi_v']:.3f}, MI(p)={r['mi_p']:.3f}")
        print(f"  MSE={r['mse']:.3f}")

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"{'dim':>4} | {'Sil':>6} | {'ARI(v)':>7} | {'MI(v)':>6} | {'MI(p)':>6} | {'MSE':>6}")
print("-"*60)

for r in results:
    if r.get("collapsed"):
        print(f"{r['dim']:>4} | {'--':>6} | {'--':>7} | {'--':>6} | {'--':>6} | {'--':>6}")
    else:
        print(f"{r['dim']:>4} | {r['sil']:>6.3f} | {r['ari_v']:>7.3f} | {r['mi_v']:>6.3f} | {r['mi_p']:>6.3f} | {r['mse']:>6.3f}")

# Trend analysis
print("\n" + "="*60)
print("TREND:")
print("="*60)

valid_results = [r for r in results if not r.get("collapsed")]
if len(valid_results) > 1:
    dims_valid = [r["dim"] for r in valid_results]
    mi_v_vals = [r["mi_v"] for r in valid_results]
    
    print(f"dim: {dims_valid}")
    print(f"MI(v): {mi_v_vals}")
    
    # Check trend
    if mi_v_vals[-1] > mi_v_vals[0]:
        print("\n>>> MI(v) INCREASES as dim decreases! <<<")
    else:
        print("\n>>> No clear trend <<<")
