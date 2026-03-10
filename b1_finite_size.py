#!/usr/bin/env python3
"""
B1: Finite-Size Scaling Experiment
Verify if phase transition is universal across system sizes

N = 32, 64, 128, 256
λ = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

np.random.seed(42)

class Ball:
    """Random acceleration ball"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * 10 + 3
        self.vel = (np.random.rand(2) - 0.5) * 0.4
        self.history = [self.pos.copy() for _ in range(5)]
        return np.array(self.history).flatten()
    
    def step(self):
        acc = (np.random.rand(2) - 0.5) * 0.1
        self.vel += acc
        self.vel = np.clip(self.vel, -0.8, 0.8)
        self.pos += self.vel
        
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)
        
        self.history.append(self.pos.copy())
        if len(self.history) > 5:
            self.history.pop(0)
        
        return np.array(self.history).flatten()


class Model:
    """Model with configurable hidden size"""
    def __init__(self, n_hidden=32, lam=0.01, lr=0.01):
        self.lam = lam
        self.lr = lr
        self.n_hidden = n_hidden
        self.W = np.random.randn(n_hidden, 10) * 0.1
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


def run_single(n_hidden, lam, seed):
    np.random.seed(seed)
    ball = Ball()
    model = Model(n_hidden=n_hidden, lam=lam)
    
    h_list, v_list = [], []
    
    for step in range(3000):
        x = ball.step()
        y = ball.vel.copy()
        model.update(x, y)
        
        if step % 100 == 0:
            h_list.append(model.h.copy())
            v_list.append(ball.vel.copy())
    
    H = np.array(h_list)
    V = np.array(v_list)
    
    pca = PCA(n_components=min(10, H.shape[1]))
    Hp = pca.fit_transform(H)
    
    sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
    
    v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
    v_bins = np.digitize(v_mag, np.percentile(v_mag, [33, 66]))
    ari = adjusted_rand_score(v_bins, KMeans(3, n_init=10).fit_predict(Hp))
    
    h_mag = np.sqrt((H**2).sum(axis=1))
    mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1]) if len(v_mag)>0 else 0
    mi_v = 0 if np.isnan(mi_v) else mi_v
    
    return {"sil": sil, "ari": ari, "mi_v": mi_v}


def run_finite_size():
    """B1: Finite-size scaling"""
    sizes = [32, 64, 128, 256]
    lams = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    seeds = [42, 123, 456]
    
    results = {}  # {n_hidden: [{lam, sil, ari, mi_v}, ...]}
    
    for n in sizes:
        print(f"\n{'='*50}")
        print(f"N = {n}")
        print(f"{'='*50}")
        
        results[n] = []
        
        for lam in lams:
            lam_r = []
            for s in seeds:
                r = run_single(n, lam, s)
                lam_r.append(r)
            
            avg = {
                "lam": lam,
                "sil": np.mean([x["sil"] for x in lam_r]),
                "ari": np.mean([x["ari"] for x in lam_r]),
                "mi_v": np.mean([x["mi_v"] for x in lam_r]),
                "sil_std": np.std([x["sil"] for x in lam_r]),
                "mi_v_std": np.std([x["mi_v"] for x in lam_r]),
            }
            results[n].append(avg)
            print(f"λ={lam:.3f}: Sil={avg['sil']:.3f}, MI(v)={avg['mi_v']:.3f}")
    
    return results


def plot_finite_size(results):
    """Plot finite-size scaling results"""
    sizes = list(results.keys())
    lams = [r["lam"] for r in results[sizes[0]]]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Silhouette vs λ for different N
    ax = axes[0, 0]
    for n in sizes:
        sil = [r["sil"] for r in results[n]]
        ax.plot(lams, sil, 'o-', label=f'N={n}', linewidth=2)
    ax.set_xlabel('λ')
    ax.set_ylabel('Silhouette')
    ax.set_title('Finite-Size Scaling: Silhouette')
    ax.legend()
    ax.grid(True)
    
    # 2. MI(v) vs λ for different N
    ax = axes[0, 1]
    for n in sizes:
        mi_v = [r["mi_v"] for r in results[n]]
        ax.plot(lams, mi_v, 'o-', label=f'N={n}', linewidth=2)
    ax.set_xlabel('λ')
    ax.set_ylabel('MI(v)')
    ax.set_title('Finite-Size Scaling: MI(v)')
    ax.legend()
    ax.grid(True)
    
    # 3. Critical λ vs N (find where MI(v) starts to dominate)
    ax = axes[1, 0]
    critical_lams = []
    for n in sizes:
        # Find first λ where MI(v) > 0.3
        mi_v_list = [r["mi_v"] for r in results[n]]
        for i, lam in enumerate(lams):
            if mi_v_list[i] > 0.3:
                critical_lams.append(lam)
                break
        else:
            critical_lams.append(lams[-1])
    
    ax.plot(sizes, critical_lams, 'o-', linewidth=2, color='red')
    ax.set_xlabel('N (system size)')
    ax.set_ylabel('Critical λ')
    ax.set_title('Critical λ vs System Size')
    ax.grid(True)
    
    # 4. Scaled Silhouette (if phase transition is universal)
    ax = axes[1, 1]
    for n in sizes:
        sil = [r["sil"] for r in results[n]]
        # Simple scaling: divide by N
        ax.plot(lams, np.array(sil), 'o-', label=f'N={n}', linewidth=2)
    ax.set_xlabel('λ')
    ax.set_ylabel('Silhouette / log(N)')
    ax.set_title('Scaled Silhouette')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("b1_finite_size_scaling.png", dpi=150)
    print("\nSaved to b1_finite_size_scaling.png")
    
    # Print summary
    print("\n" + "="*60)
    print("FINITE-SIZE SCALING SUMMARY:")
    print("="*60)
    print(f"System sizes tested: {sizes}")
    print(f"Critical λ estimates: {critical_lams}")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("B1: Finite-Size Scaling Experiment")
    print("="*60)
    results = run_finite_size()
    plot_finite_size(results)
    print("\nDone!")
