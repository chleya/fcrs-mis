#!/usr/bin/env python3
"""
FCRS-MIS V5: 长时序预测版
- 输入: 3帧位置序列
- 输出: 未来10步位置
- 强制学习速度

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
    """单球环境"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * 10 + 3
        self.vel = (np.random.rand(2) - 0.5) * 0.6
        return self.get_history(3)
    
    def step(self):
        self.pos += self.vel
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)
        return self.pos.copy()
    
    def get_history(self, n):
        """返回最近n帧的位置"""
        return self.history[-n:] if len(self.history) >= n else [self.pos.copy() for _ in range(n)]


class ModelV5:
    """V5模型: 编码3帧历史，预测10步未来"""
    def __init__(self, input_dim=6, hidden=32, predict_steps=10, lam=0.01, lr=0.01):
        self.lam = lam
        self.lr = lr
        self.predict_steps = predict_steps
        
        # 编码器: 6D(3帧x,y) -> hidden
        self.W_enc = np.random.randn(hidden, input_dim) * 0.1
        # 循环: hidden -> hidden
        self.W_rec = np.random.randn(hidden, hidden) * 0.1
        # 解码器: hidden -> 20D(10步x,y)
        self.W_dec = np.random.randn(predict_steps * 2, hidden) * 0.1
        
        self.h = np.zeros(hidden)
    
    def forward(self, history):
        """history: 3帧位置的6维向量"""
        self.h = np.tanh(self.W_enc @ history + self.W_rec @ self.h)
        pred = self.W_dec @ self.h
        return pred.reshape(self.predict_steps, 2)
    
    def update(self, history, future_seq, true_vel=None):
        """
        history: 3帧位置 (6D)
        future_seq: 未来10步位置 (10,2)
        true_vel: 真实速度 (2D) - 可选
        """
        pred = self.forward(history)
        
        # 位置预测损失
        loss_pos = np.mean((future_seq - pred) ** 2)
        
        # 速度正则 (如果提供)
        loss_vel = 0
        if true_vel is not None:
            # 从预测序列计算首步速度
            pred_vel = pred[0] - history[-1, :]  # 首步位移作为速度预测
            loss_vel = np.mean((pred_vel - true_vel) ** 2)
        
        # 压缩损失
        loss_comp = (np.mean(np.abs(self.W_enc)) + 
                   np.mean(np.abs(self.W_rec)) + 
                   np.mean(np.abs(self.W_dec))) / 3
        
        # 更新
        total_loss = loss_pos + 0.01 * loss_vel + self.lam * loss_comp
        
        # 简化梯度
        grad_enc = -np.mean(future_seq - pred) * np.mean(self.h) - self.lam * np.sign(self.W_enc)
        grad_rec = -np.mean(future_seq - pred) * np.mean(self.h) - self.lam * np.sign(self.W_rec)
        grad_dec = np.mean(future_seq - pred).reshape(-1,1) @ self.h.reshape(1,-1) - self.lam * np.sign(self.W_dec)
        
        self.W_enc += self.lr * grad_enc
        self.W_rec += self.lr * grad_rec
        self.W_dec += self.lr * grad_dec
        
        # 稀疏化
        self.W_enc[np.abs(self.W_enc) < 1e-4] = 0
        self.W_rec[np.abs(self.W_rec) < 1e-4] = 0
        self.W_dec[np.abs(self.W_dec) < 1e-4] = 0
        
        return loss_pos, loss_vel, loss_comp


def run_v5():
    """V5实验"""
    # 更强的λ测试
    lambdas = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    
    results = []
    
    for lam in lambdas:
        print(f"\n{'='*50}")
        print(f"λ = {lam}")
        print(f"{'='*50}")
        
        env = BallEnv()
        model = ModelV5(lam=lam)
        
        # 记录
        h_list, vel_list, pos_list = [], [], []
        pred_errors, vel_errors = [], []
        
        env.reset()
        
        # 训练
        for step in range(3000):
            # 获取3帧历史
            history = np.array(env.get_history(3)).flatten()  # 6D
            
            # 真实速度
            true_vel = (env.pos - env.history[-2]) if len(env.history) >= 2 else np.zeros(2)
            if not hasattr(env, 'history'):
                env.history = [env.pos.copy()]
            env.history.append(env.pos.copy())
            
            # 获取未来10步
            future = []
            temp_pos = env.pos.copy()
            temp_vel = env.vel.copy()
            for _ in range(10):
                temp_pos += temp_vel
                for i in range(2):
                    if temp_pos[i] < 1 or temp_pos[i] > 14:
                        temp_vel[i] *= -1
                        temp_pos[i] = np.clip(temp_pos[i], 1, 14)
                future.append(temp_pos.copy())
            future = np.array(future)
            
            # 更新
            env.step()
            
            mse, vel_mse, comp = model.update(history, future, true_vel)
            
            pred_errors.append(mse)
            vel_errors.append(vel_mse)
            
            if step % 100 == 0:
                # 记录隐状态
                h_list.append(model.h.copy())
                vel_list.append(true_vel.copy())
                pos_list.append(env.pos.copy())
            
            if (step + 1) % 1000 == 0:
                print(f"Step {step+1}: MSE={mse:.4f}, Vel={vel_mse:.4f}")
        
        # 评估
        H = np.array(h_list)
        V = np.array(vel_list)
        P = np.array(pos_list)
        
        # Silhouette
        pca = PCA(n_components=min(10, H.shape[1]))
        Hp = pca.fit_transform(H)
        sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
        
        # ARI (用速度方向聚类)
        vel_dir = np.arctan2(V[:, 1], V[:, 0])
        vel_bins = np.digitize(vel_dir, bins=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        true_lab = vel_bins
        pred_lab = KMeans(3, n_init=10).fit_predict(Hp)
        ari = adjusted_rand_score(true_lab, pred_lab)
        
        # MI: 隐状态 vs 速度
        v_mag = np.sqrt(V[:, 0]**2 + V[:, 1]**2)
        h_mag = np.sqrt((H**2).sum(axis=1))
        mi_v = abs(np.corrcoef(v_mag, h_mag)[0, 1]) if len(v_mag) > 0 else 0
        
        # MI: 隐状态 vs 位置
        p_mag = np.sqrt(P[:, 0]**2 + P[:, 1]**2)
        mi_p = abs(np.corrcoef(p_mag, h_mag)[0, 1]) if len(p_mag) > 0 else 0
        
        # 结构熵
        W = np.abs(model.W_enc.flatten())
        W = W[W > 1e-4]
        ent = -np.sum((W/W.sum())*np.log2(W/W.sum())) if len(W)>0 else 0
        
        print(f"\nFinal:")
        print(f"  Silhouette: {sil:.3f}")
        print(f"  ARI: {ari:.3f}")
        print(f"  MI(v): {mi_v:.3f}")
        print(f"  MI(p): {mi_p:.3f}")
        print(f"  MI(v) > MI(p): {'YES' if mi_v > mi_p else 'NO'}")
        
        results.append({
            "lambda": lam,
            "sil": sil,
            "ari": ari,
            "mi_v": mi_v,
            "mi_p": mi_p,
            "ent": ent,
            "pred_err": pred_errors[-500:],
            "vel_err": vel_errors[-500:]
        })
    
    return results


def plot_v5(results):
    """可视化"""
    lams = [r["lambda"] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Silhouette
    axes[0,0].plot(lams, [r["sil"] for r in results], 'o-')
    axes[0,0].set_title("Silhouette vs λ")
    axes[0,0].grid(True)
    
    # 2. ARI
    axes[0,1].plot(lams, [r["ari"] for r in results], 'o-', color='green')
    axes[0,1].set_title("ARI vs λ")
    axes[0,1].grid(True)
    
    # 3. MI对比
    axes[0,2].plot(lams, [r["mi_v"] for r in results], 'o-', label='MI(v)', color='blue')
    axes[0,2].plot(lams, [r["mi_p"] for r in results], 's--', label='MI(p)', color='red')
    axes[0,2].set_title("MI: Velocity vs Position")
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # 4. 结构熵
    axes[1,0].plot(lams, [r["ent"] for r in results], 'o-', color='orange')
    axes[1,0].set_title("Structural Entropy vs λ")
    axes[1,0].grid(True)
    
    # 5. 预测误差
    for r in results:
        if r["lambda"] in [0.01, 0.02, 0.05]:
            axes[1,1].plot(r["pred_err"][::20], label=f"λ={r['lambda']}")
    axes[1,1].set_title("Prediction Error")
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # 6. MI对比条形图
    x = np.arange(len(lams))
    width = 0.35
    axes[1,2].bar(x - width/2, [r["mi_v"] for r in results], width, label='MI(v)')
    axes[1,2].bar(x + width/2, [r["mi_p"] for r in results], width, label='MI(p)')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels([str(l) for l in lams])
    axes[1,2].set_title("MI Comparison")
    axes[1,2].legend()
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig("fcrs_mis_v5_results.png", dpi=150)
    print("\nSaved to fcrs_mis_v5_results.png")
    
    # 打印验证结果
    print("\n" + "="*50)
    print("Verification:")
    print("="*50)
    for r in results:
        status = "OK" if r["mi_v"] > r["mi_p"] else "NO"
        print(f"λ={r['lambda']:.3f}: MI(v)={r['mi_v']:.3f}, MI(p)={r['mi_p']:.3f} [{status}]")
    print("="*50)


if __name__ == "__main__":
    print("FCRS-MIS V5: Long-term Prediction")
    print("="*50)
    results = run_v5()
    plot_v5(results)
    print("\nDone!")
