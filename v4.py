#!/usr/bin/env python3
"""
FCRS-MIS V4: 补全核心验证
- MI(速度) vs MI(观测) 直接对比
- 预测误差曲线
- 更长训练

Author: NeuralSite Team
Date: 2026-03-10
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

np.random.seed(42)

class BallEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * 10 + 3
        self.vel = (np.random.rand(2) - 0.5) * 0.6
        return self.state()
    
    def step(self):
        self.pos += self.vel
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)
        return self.state()
    
    def state(self):
        return np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1]])


class Model:
    def __init__(self, n_hidden=32, lam=0.01, lr=0.01):
        self.lam = lam
        self.lr = lr
        self.W = np.random.randn(n_hidden, 4) * 0.1
        self.h = np.zeros(n_hidden)
    
    def forward(self, s):
        self.h = np.tanh(self.W @ s)
        return self.W.T @ self.h
    
    def update(self, s, sn):
        p = self.forward(s)
        e = sn - p
        mse = np.mean(e**2)
        comp = np.mean(np.abs(self.W))
        
        # Update
        self.W += self.lr * (np.mean(e) * np.mean(self.h) - self.lam * np.sign(self.W))
        self.W[np.abs(self.W) < 1e-4] = 0
        
        return mse, comp, np.mean(np.abs(self.h))


def compute_mi(state_list, h_list):
    """计算两个MI: 状态vs速度, 状态vs位置"""
    if len(state_list) < 10:
        return 0, 0, 0, 0
    
    S = np.array(state_list)
    H = np.array(h_list)
    
    # 速度 magnitude
    v_mag = np.sqrt(S[1:, 2]**2 + S[1:, 3]**2)
    # 位置 magnitude  
    p_mag = np.sqrt(S[1:, 0]**2 + S[1:, 1]**2)
    # 状态变化
    h_mag = np.sqrt((H[1:]**2).sum(axis=1))
    
    # MI with velocity
    if len(v_mag) > 0 and len(h_mag) > 0:
        corr_v = np.corrcoef(v_mag[:len(h_mag)], h_mag)[0, 1]
        mi_v = abs(corr_v) if not np.isnan(corr_v) else 0
    else:
        mi_v = 0
    
    # MI with position
    if len(p_mag) > 0 and len(h_mag) > 0:
        corr_p = np.corrcoef(p_mag[:len(h_mag)], h_mag)[0, 1]
        mi_p = abs(corr_p) if not np.isnan(corr_p) else 0
    else:
        mi_p = 0
    
    return mi_v, mi_p, v_mag.mean(), p_mag.mean()


def run_v4():
    """V4核心验证版"""
    # 细粒度λ测试
    lambdas = [0, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02]
    
    results = []
    
    for lam in lambdas:
        print(f"\n{'='*50}")
        print(f"λ = {lam}")
        print(f"{'='*50}")
        
        env = BallEnv()
        model = Model(n_hidden=32, lam=lam, lr=0.01)
        
        # 记录
        h_list, state_list = [], []
        pred_errors = []
        compress_errors = []
        activations = []
        
        env.reset()
        
        # 训练3000步（更长）
        for step in range(3000):
            s = env.state()
            sn = env.step()
            
            mse, comp, act = model.update(s, sn)
            
            pred_errors.append(mse)
            compress_errors.append(comp)
            activations.append(act)
            
            if step % 100 == 0:
                h_list.append(model.h.copy())
                state_list.append(s.copy())
            
            if (step + 1) % 1000 == 0:
                print(f"Step {step+1}: MSE={mse:.4f}, Comp={comp:.4f}, Act={act:.4f}")
        
        # 评估指标
        H = np.array(h_list)
        S = np.array(state_list)
        
        # Silhouette
        pca = PCA(n_components=min(10, H.shape[1]))
        Hp = pca.fit_transform(H)
        sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
        
        # ARI (用位置聚类作为真实标签)
        true_labels = KMeans(3, n_init=10).fit_predict(S[:, :2])
        pred_labels = KMeans(3, n_init=10).fit_predict(Hp)
        ari = adjusted_rand_score(true_labels, pred_labels)
        
        # MI计算
        mi_v, mi_p, _, _ = compute_mi(state_list, h_list)
        
        # 结构熵
        W = np.abs(model.W.flatten())
        W = W[W > 1e-4]
        ent = -np.sum((W/W.sum())*np.log2(W/W.sum())) if len(W)>0 else 0
        
        # 最终预测误差（后500步平均）
        final_mse = np.mean(pred_errors[-500:])
        
        print(f"\nFinal Metrics:")
        print(f"  Silhouette: {sil:.3f}")
        print(f"  ARI: {ari:.3f}")
        print(f"  MI(velocity): {mi_v:.3f}")
        print(f"  MI(position): {mi_p:.3f}")
        print(f"  MI(v) > MI(p): {'OK' if mi_v > mi_p else 'NO'}")
        print(f"  Final MSE: {final_mse:.3f}")
        print(f"  Entropy: {ent:.3f}")
        
        results.append({
            "lambda": lam,
            "sil": sil,
            "ari": ari,
            "mi_v": mi_v,
            "mi_p": mi_p,
            "ent": ent,
            "final_mse": final_mse,
            "pred_errors": pred_errors,
            "activations": activations
        })
    
    return results


def plot_v4(results):
    """V4可视化"""
    lams = [r["lambda"] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Silhouette
    ax = axes[0, 0]
    ax.plot(lams, [r["sil"] for r in results], 'o-', linewidth=2)
    ax.set_title("Silhouette vs λ", fontsize=14)
    ax.set_xlabel("λ")
    ax.grid(True)
    
    # 2. ARI
    ax = axes[0, 1]
    ax.plot(lams, [r["ari"] for r in results], 'o-', linewidth=2, color='green')
    ax.set_title("ARI vs λ", fontsize=14)
    ax.set_xlabel("λ")
    ax.grid(True)
    
    # 3. MI对比 (核心验证!)
    ax = axes[0, 2]
    ax.plot(lams, [r["mi_v"] for r in results], 'o-', linewidth=2, label='MI(状态 vs 速度)', color='blue')
    ax.plot(lams, [r["mi_p"] for r in results], 's--', linewidth=2, label='MI(状态 vs 位置)', color='red')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("MI对比 (核心验证)", fontsize=14)
    ax.set_xlabel("λ")
    ax.legend()
    ax.grid(True)
    
    # 4. 结构熵
    ax = axes[1, 0]
    ax.plot(lams, [r["ent"] for r in results], 'o-', linewidth=2, color='orange')
    ax.set_title("Structural Entropy vs λ", fontsize=14)
    ax.set_xlabel("λ")
    ax.grid(True)
    
    # 5. 预测误差
    ax = axes[1, 1]
    for r in results:
        if r["lambda"] in [0, 0.005, 0.01]:
            ax.plot(r["pred_errors"][::50], label=f"λ={r['lambda']}", alpha=0.7)
    ax.set_title("Prediction Error Curves", fontsize=14)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True)
    
    # 6. 最终MSE对比
    ax = axes[1, 2]
    ax.plot(lams, [r["final_mse"] for r in results], 'o-', linewidth=2, color='purple')
    ax.set_title("Final MSE vs λ", fontsize=14)
    ax.set_xlabel("λ")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("fcrs_mis_v4_results.png", dpi=150)
    print("\nSaved to fcrs_mis_v4_results.png")
    
    # 打印关键验证
    print("\n" + "="*60)
    print("核心验证结果:")
    print("="*60)
    for r in results:
        status = "OK" if r["mi_v"] > r["mi_p"] else "NO"
        print(f"λ={r['lambda']:.3f}: MI(v)={r['mi_v']:.3f}, MI(p)={r['mi_p']:.3f} {status}")
    print("="*60)


if __name__ == "__main__":
    print("FCRS-MIS V4: 核心验证版")
    print("="*60)
    results = run_v4()
    plot_v4(results)
    print("\nDone!")
