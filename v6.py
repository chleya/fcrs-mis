#!/usr/bin/env python3
"""
FCRS-MIS V6: 临界补全版
- 细粒度λ: 0.03-0.06
- 3次独立重复
- ARI用速度标签

Author: NeuralSite Team
Date: 2026-03-10
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

np.random.seed(42)

class Ball:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * 10 + 3
        self.vel = (np.random.rand(2) - 0.5) * 0.6
        self.history = [self.pos.copy(), self.pos.copy(), self.pos.copy()]
        return np.array(self.history).flatten()
    
    def step(self):
        self.pos += self.vel
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)
        self.history.append(self.pos.copy())
        if len(self.history) > 3:
            self.history.pop(0)
        return np.array(self.history).flatten()


class Model:
    def __init__(self, hidden=32, lam=0.01, lr=0.01):
        self.lam = lam
        self.lr = lr
        self.W = np.random.randn(hidden, 6) * 0.1
        self.h = np.zeros(hidden)
    
    def forward(self, x):
        self.h = np.tanh(self.W @ x)
        pred_pos = self.W.T @ self.h
        return pred_pos[:2]
    
    def update(self, x, y):
        p = self.forward(x)
        e = y - p
        mse = np.mean(e**2)
        comp = np.mean(np.abs(self.W))
        delta = np.mean(e) * np.mean(self.h) - self.lam * np.sign(self.W)
        self.W += self.lr * delta
        self.W[np.abs(self.W) < 1e-4] = 0
        return mse, comp


def run_single(lam, seed):
    """单次实验"""
    np.random.seed(seed)
    
    ball = Ball()
    model = Model(lam=lam)
    
    h_list, v_list = [], []
    
    for step in range(3000):
        x = ball.reset()
        y = ball.pos.flatten()
        mse, comp = model.update(x, y)
        
        if step % 100 == 0:
            h_list.append(model.h.copy())
            v_list.append(ball.vel.copy())
    
    H = np.array(h_list)
    V = np.array(v_list)
    
    # PCA
    pca = PCA(n_components=min(10, H.shape[1]))
    Hp = pca.fit_transform(H)
    
    # Silhouette
    sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
    
    # ARI - 用速度大小标签 (低速/中速/高速)
    v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
    v_bins = np.digitize(v_mag, bins=[0.1, 0.2, 0.3])  # 三分
    true_lab = v_bins
    pred_lab = KMeans(3, n_init=10).fit_predict(Hp)
    ari = adjusted_rand_score(true_lab, pred_lab)
    
    # MI
    h_mag = np.sqrt((H**2).sum(axis=1))
    mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1]) if len(v_mag)>0 else 0
    
    # MI vs position
    p_list = [ball.pos.copy() for _ in range(len(h_list))]
    P = np.array(p_list)
    p_mag = np.sqrt(P[:,0]**2 + P[:,1]**2)
    mi_p = abs(np.corrcoef(p_mag, h_mag)[0,1]) if len(p_mag)>0 else 0
    
    # Entropy
    W = np.abs(model.W.flatten())
    W = W[W>1e-4]
    ent = -np.sum((W/W.sum())*np.log2(W/W.sum())) if len(W)>0 else 0
    
    return {
        "sil": sil,
        "ari": ari,
        "mi_v": mi_v,
        "mi_p": mi_p,
        "ent": ent
    }


def run_v6():
    """V6实验: 细粒度λ + 3次重复"""
    lambdas = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
    seeds = [42, 123, 456]  # 3次独立
    
    all_results = []
    
    for lam in lambdas:
        print(f"\n{'='*50}")
        print(f"λ = {lam}")
        print(f"{'='*50}")
        
        lam_results = []
        for i, seed in enumerate(seeds):
            r = run_single(lam, seed)
            lam_results.append(r)
            print(f"  Run {i+1}: Sil={r['sil']:.3f}, ARI={r['ari']:.3f}, MI(v)={r['mi_v']:.3f}, MI(p)={r['mi_p']:.3f}")
        
        # 平均
        avg = {
            "lambda": lam,
            "sil": np.mean([x["sil"] for x in lam_results]),
            "ari": np.mean([x["ari"] for x in lam_results]),
            "mi_v": np.mean([x["mi_v"] for x in lam_results]),
            "mi_p": np.mean([x["mi_p"] for x in lam_results]),
            "ent": np.mean([x["ent"] for x in lam_results]),
            "mi_v_std": np.std([x["mi_v"] for x in lam_results]),
            "mi_p_std": np.std([x["mi_p"] for x in lam_results]),
        }
        all_results.append(avg)
        print(f"  AVG: Sil={avg['sil']:.3f}, ARI={avg['ari']:.3f}, MI(v)={avg['mi_v']:.3f}, MI(p)={avg['mi_p']:.3f}")
    
    return all_results


def plot_v6(results):
    """可视化"""
    lams = [r["lambda"] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Silhouette
    axes[0,0].plot(lams, [r["sil"] for r in results], 'o-', linewidth=2)
    axes[0,0].set_title("Silhouette vs λ")
    axes[0,0].grid(True)
    
    # 2. ARI (velocity-based label)
    axes[0,1].plot(lams, [r["ari"] for r in results], 'o-', linewidth=2, color='green')
    axes[0,1].set_title("ARI (velocity label)")
    axes[0,1].grid(True)
    
    # 3. MI对比
    axes[0,2].errorbar(lams, [r["mi_v"] for r in results], yerr=[r["mi_v_std"] for r in results], 
                       fmt='o-', label='MI(v)', color='blue', capsize=3)
    axes[0,2].errorbar(lams, [r["mi_p"] for r in results], yerr=[r["mi_p_std"] for r in results],
                       fmt='s--', label='MI(p)', color='red', capsize=3)
    axes[0,2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0,2].set_title("MI Comparison (with std)")
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # 4. 结构熵
    axes[1,0].plot(lams, [r["ent"] for r in results], 'o-', linewidth=2, color='orange')
    axes[1,0].set_title("Structural Entropy")
    axes[1,0].grid(True)
    
    # 5. MI差距 (MI(v) - MI(p))
    mi_diff = [r["mi_v"] - r["mi_p"] for r in results]
    axes[1,1].plot(lams, mi_diff, 'o-', linewidth=2, color='purple')
    axes[1,1].axhline(y=0, color='red', linestyle='--')
    axes[1,1].fill_between(lams, mi_diff, 0, where=[x>0 for x in mi_diff], alpha=0.3, color='green')
    axes[1,1].fill_between(lams, mi_diff, 0, where=[x<0 for x in mi_diff], alpha=0.3, color='red')
    axes[1,1].set_title("MI(v) - MI(p)")
    axes[1,1].grid(True)
    
    # 6. ARI提升 (相对于λ=0.03基线)
    base_ari = results[0]["ari"]
    ari提升 = [(r["ari"] - base_ari) / (abs(base_ari)+1e-6) * 100 for r in results]
    axes[1,2].plot(lams, ari提升, 'o-', linewidth=2, color='brown')
    axes[1,2].axhline(y=0, color='gray', linestyle='--')
    axes[1,2].set_title("ARI Improvement %")
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig("fcrs_mis_v6_results.png", dpi=150)
    print("\nSaved to fcrs_mis_v6_results.png")
    
    # 验证
    print("\n" + "="*60)
    print("Verification (averaged over 3 runs):")
    print("="*60)
    for r in results:
        status = "OK" if r["mi_v"] > r["mi_p"] else "NO"
        diff = r["mi_v"] - r["mi_p"]
        print(f"λ={r['lambda']:.3f}: MI(v)={r['mi_v']:.3f}±{r['mi_v_std']:.3f}, MI(p)={r['mi_p']:.3f}±{r['mi_p_std']:.3f}, diff={diff:+.3f} [{status}]")
    print("="*60)


if __name__ == "__main__":
    print("FCRS-MIS V6: Critical Zone")
    print("="*60)
    results = run_v6()
    plot_v6(results)
    print("\nDone!")
