#!/usr/bin/env python3
"""FCRS-MIS V6.1: Velocity Prediction - Force learning dynamics"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

np.random.seed(42)

class Ball:
    def __init__(self):
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.history = []
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
        # 输出速度 (2D)
        return (self.W.T @ self.h)[:2]
    
    def update(self, x, y):
        p = self.forward(x)
        e = y - p
        mse = np.mean(e**2)
        delta = np.mean(e) * np.mean(self.h) - self.lam * np.sign(self.W)
        self.W += self.lr * delta
        self.W[np.abs(self.W) < 1e-4] = 0
        return mse


def run_single(lam, seed):
    np.random.seed(seed)
    ball = Ball()
    model = Model(lam=lam)
    
    h_list, v_list, p_list = [], [], []
    
    for step in range(3000):
        # 输入: 3帧位置
        x = ball.reset()
        # 目标: 当前速度 (2D) - 强制学动力学!
        y = ball.vel.copy()
        
        model.update(x, y)
        
        if step % 100 == 0:
            h_list.append(model.h.copy())
            v_list.append(ball.vel.copy())
            p_list.append(ball.pos.copy())
    
    H = np.array(h_list)
    V = np.array(v_list)
    P = np.array(p_list)
    
    pca = PCA(n_components=min(10, H.shape[1]))
    Hp = pca.fit_transform(H)
    
    sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
    
    # ARI with velocity magnitude
    v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
    v_bins = np.digitize(v_mag, [0.1, 0.2, 0.3])
    ari = adjusted_rand_score(v_bins, KMeans(3, n_init=10).fit_predict(Hp))
    
    # MI: h vs velocity
    h_mag = np.sqrt((H**2).sum(axis=1))
    mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1]) if len(v_mag)>0 else 0
    mi_v = 0 if np.isnan(mi_v) else mi_v
    
    # MI: h vs position
    p_mag = np.sqrt(P[:,0]**2 + P[:,1]**2)
    mi_p = abs(np.corrcoef(p_mag, h_mag)[0,1]) if len(p_mag)>0 else 0
    mi_p = 0 if np.isnan(mi_p) else mi_p
    
    # Entropy
    W = np.abs(model.W.flatten())
    W = W[W>1e-4]
    ent = -np.sum((W/W.sum())*np.log2(W/W.sum())) if len(W)>0 else 0
    
    return {"sil": sil, "ari": ari, "mi_v": mi_v, "mi_p": mi_p, "ent": ent}


def run():
    lams = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    seeds = [42, 123, 456]
    results = []
    
    for lam in lams:
        print(f"\n=== λ = {lam} ===")
        lam_r = []
        for i, s in enumerate(seeds):
            r = run_single(lam, s)
            lam_r.append(r)
            print(f"  Run {i+1}: Sil={r['sil']:.3f}, ARI={r['ari']:.3f}, MI(v)={r['mi_v']:.3f}, MI(p)={r['mi_p']:.3f}")
        
        avg = {
            "lam": lam,
            "sil": np.mean([x["sil"] for x in lam_r]),
            "ari": np.mean([x["ari"] for x in lam_r]),
            "mi_v": np.mean([x["mi_v"] for x in lam_r]),
            "mi_p": np.mean([x["mi_p"] for x in lam_r]),
            "ent": np.mean([x["ent"] for x in lam_r]),
            "mi_v_std": np.std([x["mi_v"] for x in lam_r]),
            "mi_p_std": np.std([x["mi_p"] for x in lam_r]),
        }
        results.append(avg)
        print(f"  AVG: MI(v)={avg['mi_v']:.3f}, MI(p)={avg['mi_p']:.3f}")
    
    return results


def plot(results):
    lams = [r["lam"] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2,3,1)
    plt.plot(lams, [r["sil"] for r in results], 'o-')
    plt.title("Silhouette"); plt.grid(True)
    
    plt.subplot(2,3,2)
    plt.plot(lams, [r["ari"] for r in results], 'o-', color='green')
    plt.title("ARI (velocity label)"); plt.grid(True)
    
    plt.subplot(2,3,3)
    plt.errorbar(lams, [r["mi_v"] for r in results], yerr=[r["mi_v_std"] for r in results], fmt='o-', label='MI(v)', capsize=3, color='blue')
    plt.errorbar(lams, [r["mi_p"] for r in results], yerr=[r["mi_p_std"] for r in results], fmt='s--', label='MI(p)', capsize=3, color='red')
    plt.title("MI Comparison"); plt.legend(); plt.grid(True)
    
    plt.subplot(2,3,4)
    plt.plot(lams, [r["ent"] for r in results], 'o-', color='orange')
    plt.title("Entropy"); plt.grid(True)
    
    plt.subplot(2,3,5)
    diff = [r["mi_v"]-r["mi_p"] for r in results]
    plt.plot(lams, diff, 'o-', color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.fill_between(lams, diff, 0, where=[x>0 for x in diff], alpha=0.3, color='green')
    plt.title("MI(v) - MI(p)"); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("fcrs_mis_v61.png", dpi=150)
    print("\nSaved!")
    
    print("\n" + "="*60)
    print("Verification:")
    print("="*60)
    for r in results:
        status = "OK" if r["mi_v"] > r["mi_p"] else "NO"
        print(f"λ={r['lam']:.3f}: MI(v)={r['mi_v']:.3f}, MI(p)={r['mi_p']:.3f} [{status}]")
    print("="*60)


if __name__ == "__main__":
    print("V6.1 Running: Velocity Prediction")
    r = run()
    plot(r)
    print("Done!")
