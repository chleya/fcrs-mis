#!/usr/bin/env python3
"""FCRS-MIS V6.2: Random Acceleration + Long-term Prediction
The definitive experiment: force velocity learning"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

np.random.seed(42)

class Ball:
    """Ball with random acceleration"""
    def __init__(self):
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.history = []
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * 10 + 3
        self.vel = (np.random.rand(2) - 0.5) * 0.4
        self.history = [self.pos.copy() for _ in range(5)]
        return np.array(self.history).flatten()  # 5帧历史
    
    def step(self):
        # 随机加速度: v(t+1) = v(t) + ε
        acc = (np.random.rand(2) - 0.5) * 0.1  # 随机扰动
        self.vel += acc
        self.vel = np.clip(self.vel, -0.8, 0.8)  # 限制速度
        
        self.pos += self.vel
        
        # 边界反弹
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)
        
        self.history.append(self.pos.copy())
        if len(self.history) > 5:
            self.history.pop(0)
        
        return np.array(self.history).flatten()


class Model:
    def __init__(self, hidden=32, lam=0.01, lr=0.01):
        self.lam = lam
        self.lr = lr
        # 输入: 5帧位置 (10D), 输出: 20步速度 (40D)
        self.W = np.random.randn(hidden, 10) * 0.1
        self.h = np.zeros(hidden)
    
    def forward(self, x):
        self.h = np.tanh(self.W @ x)
        # 输出未来20步的速度
        return np.tile(self.W.T @ self.h, 20)[:40]  # 40D = 20步*2
    
    def update(self, x, y):
        p = self.forward(x)
        e = y - p
        mse = np.mean(e**2)
        
        # 简化梯度
        delta = np.mean(e[:2]) * np.mean(self.h) - self.lam * np.sign(self.W)
        self.W += self.lr * delta
        self.W[np.abs(self.W) < 1e-4] = 0
        
        return mse


def run_single(lam, seed):
    np.random.seed(seed)
    ball = Ball()
    model = Model(lam=lam)
    
    h_list, v_list, p_list = [], [], []
    
    for step in range(3000):
        x = ball.step()  # 获取5帧历史
        
        # 目标: 未来20步的速度
        future_vels = []
        temp_vel = ball.vel.copy()
        temp_pos = ball.pos.copy()
        for _ in range(20):
            acc = (np.random.rand(2) - 0.5) * 0.1
            temp_vel += acc
            temp_vel = np.clip(temp_vel, -0.8, 0.8)
            temp_pos += temp_vel
            for i in range(2):
                if temp_pos[i] < 1 or temp_pos[i] > 14:
                    temp_vel[i] *= -1
                    temp_pos[i] = np.clip(temp_pos[i], 1, 14)
            future_vels.append(temp_vel.copy())
        y = np.array(future_vels).flatten()
        
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
    
    # ARI: velocity magnitude
    v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
    v_bins = np.digitize(v_mag, np.percentile(v_mag, [33, 66]))
    ari = adjusted_rand_score(v_bins, KMeans(3, n_init=10).fit_predict(Hp))
    
    # MI
    h_mag = np.sqrt((H**2).sum(axis=1))
    mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1]) if len(v_mag)>0 else 0
    mi_v = 0 if np.isnan(mi_v) else mi_v
    
    p_mag = np.sqrt(P[:,0]**2 + P[:,1]**2)
    mi_p = abs(np.corrcoef(p_mag, h_mag)[0,1]) if len(p_mag)>0 else 0
    mi_p = 0 if np.isnan(mi_p) else mi_p
    
    # Entropy
    W = np.abs(model.W.flatten())
    W = W[W>1e-4]
    ent = -np.sum((W/W.sum())*np.log2(W/W.sum())) if len(W)>0 else 0
    
    return {"sil": sil, "ari": ari, "mi_v": mi_v, "mi_p": mi_p, "ent": ent}


def run():
    lams = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1]
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
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Silhouette
    axes[0,0].plot(lams, [r["sil"] for r in results], 'o-', linewidth=2)
    axes[0,0].set_title("Silhouette vs λ")
    axes[0,0].grid(True)
    
    # 2. ARI
    axes[0,1].plot(lams, [r["ari"] for r in results], 'o-', linewidth=2, color='green')
    axes[0,1].set_title("ARI (velocity label)")
    axes[0,1].grid(True)
    
    # 3. MI Comparison
    axes[0,2].errorbar(lams, [r["mi_v"] for r in results], yerr=[r["mi_v_std"] for r in results], 
                       fmt='o-', label='MI(v)', capsize=3, color='blue', linewidth=2)
    axes[0,2].errorbar(lams, [r["mi_p"] for r in results], yerr=[r["mi_p_std"] for r in results],
                       fmt='s--', label='MI(p)', capsize=3, color='red', linewidth=2)
    axes[0,2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0,2].set_title("MI: Velocity vs Position")
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # 4. Entropy
    axes[1,0].plot(lams, [r["ent"] for r in results], 'o-', linewidth=2, color='orange')
    axes[1,0].set_title("Structural Entropy")
    axes[1,0].grid(True)
    
    # 5. MI difference
    diff = [r["mi_v"]-r["mi_p"] for r in results]
    axes[1,1].plot(lams, diff, 'o-', linewidth=2, color='purple')
    axes[1,1].axhline(y=0, color='red', linestyle='--')
    axes[1,1].fill_between(lams, diff, 0, where=[x>0 for x in diff], alpha=0.3, color='green')
    axes[1,1].set_title("MI(v) - MI(p)")
    axes[1,1].grid(True)
    
    # 6. Summary bar
    x = np.arange(len(lams))
    width = 0.35
    axes[1,2].bar(x - width/2, [r["mi_v"] for r in results], width, label='MI(v)', color='blue', alpha=0.7)
    axes[1,2].bar(x + width/2, [r["mi_p"] for r in results], width, label='MI(p)', color='red', alpha=0.7)
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels([str(l) for l in lams])
    axes[1,2].set_title("MI Comparison")
    axes[1,2].legend()
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig("fcrs_mis_v62.png", dpi=150)
    print("\nSaved to fcrs_mis_v62.png")
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION RESULTS:")
    print("="*60)
    for r in results:
        status = "OK" if r["mi_v"] > r["mi_p"] else "NO"
        diff = r["mi_v"] - r["mi_p"]
        print(f"λ={r['lam']:.3f}: MI(v)={r['mi_v']:.3f}, MI(p)={r['mi_p']:.3f}, diff={diff:+.3f} [{status}]")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("V6.2: Random Acceleration + 20-step Prediction")
    print("DEFINITIVE EXPERIMENT")
    print("="*60)
    r = run()
    plot(r)
    print("\nDone!")
