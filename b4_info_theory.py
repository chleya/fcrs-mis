#!/usr/bin/env python3
"""
B4: Information-Theoretic Analysis of Critical λ
Analyze the physical meaning of critical λ from information theory perspective
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score

np.random.seed(42)

class Ball:
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
    def __init__(self, n_hidden=64, lam=0.01, lr=0.01):
        self.lam = lam
        self.lr = lr
        self.n_hidden = n_hidden
        self.W = np.random.randn(n_hidden, 10) * 0.1
        self.h = np.zeros(n_hidden)
        
        # Track information metrics
        self.entropy_history = []
        self.compression_history = []
        self.info_bottleneck_history = []
    
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
        
        # Compute information metrics
        # 1. Entropy of hidden state distribution
        h_bin = (self.h > 0).astype(int)
        h_str = ''.join(h_bin.astype(str))
        if len(set(h_str)) > 0:
            # Simple entropy estimate
            p1 = np.mean(h_bin)
            p1 = np.clip(p1, 1e-10, 1-1e-10)
            entropy = -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)
        else:
            entropy = 0
        self.entropy_history.append(entropy)
        
        # 2. Compression ratio
        n_nonzero = np.sum(np.abs(self.W) > 1e-4)
        total = self.W.size
        compression = 1 - n_nonzero / total
        self.compression_history.append(compression)
        
        # 3. Info bottleneck: I(x;h) - I(y;h)
        # Approximate with correlation
        x_mag = np.sqrt((x**2).sum())
        h_mag = np.sqrt((self.h**2).sum())
        ixh = abs(np.corrcoef(x_mag, h_mag)[0,1]) if x_mag > 0 else 0
        iyh = abs(np.corrcoef(np.sqrt((y**2).sum()), h_mag)[0,1]) if np.sqrt((y**2).sum()) > 0 else 0
        ib = iyh - ixh  # Positive means more info about output than input
        self.info_bottleneck_history.append(ib)
        
        return mse


def run_info_theory(lam, seed=42):
    """Run experiment and collect information-theoretic metrics"""
    np.random.seed(seed)
    ball = Ball()
    model = Model(n_hidden=64, lam=lam)
    
    h_list, v_list, p_list = [], [], []
    
    for step in range(3000):
        x = ball.step()
        y = ball.vel.copy()
        model.update(x, y)
        
        if step % 100 == 0:
            h_list.append(model.h.copy())
            v_list.append(ball.vel.copy())
            p_list.append(ball.pos.copy())
    
    H = np.array(h_list)
    V = np.array(v_list)
    P = np.array(p_list)
    
    # Final metrics
    pca = PCA(n_components=min(10, H.shape[1]))
    Hp = pca.fit_transform(H)
    sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
    
    v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
    h_mag = np.sqrt((H**2).sum(axis=1))
    mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1])
    mi_v = 0 if np.isnan(mi_v) else mi_v
    
    p_mag = np.sqrt(P[:,0]**2 + P[:,1]**2)
    mi_p = abs(np.corrcoef(p_mag, h_mag)[0,1])
    mi_p = 0 if np.isnan(mi_p) else mi_p
    
    # Average information metrics (last 1000 steps)
    avg_entropy = np.mean(model.entropy_history[-1000:])
    avg_compression = np.mean(model.compression_history[-1000:])
    avg_ib = np.mean(model.info_bottleneck_history[-1000:])
    
    return {
        "lam": lam,
        "sil": sil,
        "mi_v": mi_v,
        "mi_p": mi_p,
        "entropy": avg_entropy,
        "compression": avg_compression,
        "info_bottleneck": avg_ib,
    }


def run_b4():
    """B4: Information-theoretic analysis"""
    lams = [0, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1]
    
    results = []
    for lam in lams:
        print(f"λ = {lam:.3f}")
        r = run_info_theory(lam)
        results.append(r)
        print(f"  Sil={r['sil']:.3f}, MI(v)={r['mi_v']:.3f}, Entropy={r['entropy']:.3f}, IB={r['info_bottleneck']:.3f}")
    
    return results


def plot_b4(results):
    """Plot information-theoretic analysis"""
    lams = [r["lam"] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Silhouette vs λ
    axes[0, 0].plot(lams, [r["sil"] for r in results], 'o-', linewidth=2)
    axes[0, 0].set_xlabel('λ')
    axes[0, 0].set_ylabel('Silhouette')
    axes[0, 0].set_title('Order Parameter (Silhouette)')
    axes[0, 0].grid(True)
    axes[0, 0].axvline(x=0.01, color='red', linestyle='--', alpha=0.5, label='λ_c ≈ 0.01')
    axes[0, 0].legend()
    
    # 2. MI(v) vs λ
    axes[0, 1].plot(lams, [r["mi_v"] for r in results], 'o-', linewidth=2, label='MI(v)')
    axes[0, 1].plot(lams, [r["mi_p"] for r in results], 's--', linewidth=2, label='MI(p)')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('λ')
    axes[0, 1].set_ylabel('Mutual Information')
    axes[0, 1].set_title('Causal Encoding: MI(v) vs MI(p)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Entropy vs λ
    axes[0, 2].plot(lams, [r["entropy"] for r in results], 'o-', linewidth=2, color='green')
    axes[0, 2].set_xlabel('λ')
    axes[0, 2].set_ylabel('Entropy')
    axes[0, 2].set_title('Hidden State Entropy')
    axes[0, 2].grid(True)
    
    # 4. Compression vs λ
    axes[1, 0].plot(lams, [r["compression"] for r in results], 'o-', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('λ')
    axes[1, 0].set_ylabel('Compression Ratio')
    axes[1, 0].set_title('Weight Sparsity')
    axes[1, 0].grid(True)
    
    # 5. Info Bottleneck vs λ
    axes[1, 1].plot(lams, [r["info_bottleneck"] for r in results], 'o-', linewidth=2, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('λ')
    axes[1, 1].set_ylabel('I(y;h) - I(x;h)')
    axes[1, 1].set_title('Information Bottleneck')
    axes[1, 1].grid(True)
    
    # 6. Phase diagram
    axes[1, 2].scatter(lams, [r["mi_v"] - r["mi_p"] for r in results], 
                       c=[r["sil"] for r in results], s=100, cmap='RdYlGn')
    axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('λ')
    axes[1, 2].set_ylabel('MI(v) - MI(p)')
    axes[1, 2].set_title('Phase Diagram\n(color = Silhouette)')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig("b4_info_theory.png", dpi=150)
    print("\nSaved to b4_info_theory.png")
    
    # Print key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    
    # Find critical λ
    for i, r in enumerate(results):
        if r["mi_v"] > r["mi_p"]:
            print(f"Critical λ (MI(v) > MI(p)): {r['lam']}")
            print(f"  At λ={r['lam']}: Entropy={r['entropy']:.3f}, Compression={r['compression']:.3f}")
            break
    
    # Correlation analysis
    print("\nCorrelations:")
    sil_vals = [r["sil"] for r in results]
    comp_vals = [r["compression"] for r in results]
    ib_vals = [r["info_bottleneck"] for r in results]
    
    print(f"  Sil vs Compression: {np.corrcoef(sil_vals, comp_vals)[0,1]:.3f}")
    print(f"  Sil vs InfoBottleneck: {np.corrcoef(sil_vals, ib_vals)[0,1]:.3f}")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("B4: Information-Theoretic Analysis")
    print("="*60)
    results = run_b4()
    plot_b4(results)
    print("\nDone!")
