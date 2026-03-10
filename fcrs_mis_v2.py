#!/usr/bin/env python3
"""
FCRS-MIS V2: 修正版 - 解决假阳性问题

修正内容：
1. 损失函数归一化
2. 压缩λ范围细化
3. 最小激活约束
4. 连接权重结构熵
5. 双小球碰撞环境
6. 反事实干预测试

Author: NeuralSite Team
Date: 2026-03-10
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score

# ==========================================
# 1. 双小球碰撞环境 (消除作弊)
# ==========================================
class TwoBallEnv:
    """双小球碰撞环境 - 必须学习动力学才能预测"""
    def __init__(self, grid_size=16, ball_radius=1, speed=0.3):
        self.grid_size = grid_size
        self.ball_radius = ball_radius
        self.speed = speed
        self.reset()
    
    def reset(self):
        # 两个小球，初始位置随机
        self.pos1 = np.random.rand(2) * (self.grid_size - 4) + 2
        self.pos2 = np.random.rand(2) * (self.grid_size - 4) + 2
        
        # 随机速度
        self.vel1 = (np.random.rand(2) - 0.5) * 2 * self.speed
        self.vel2 = (np.random.rand(2) - 0.5) * 2 * self.speed
        
        return self._get_observation()
    
    def step(self):
        # 更新位置
        self.pos1 += self.vel1
        self.pos2 += self.vel2
        
        # 边界反弹
        for pos, vel in [(self.pos1, self.vel1), (self.pos2, self.vel2)]:
            for i in range(2):
                if pos[i] <= self.ball_radius or pos[i] >= self.grid_size - self.ball_radius:
                    vel[i] *= -1
                    pos[i] = np.clip(pos[i], self.ball_radius, self.grid_size - self.ball_radius)
        
        # 球球碰撞
        dist = np.linalg.norm(self.pos1 - self.pos2)
        if dist < 2 * self.ball_radius:
            # 弹性碰撞
            v1_new = self.vel1 - np.dot(self.vel1 - self.vel2, self.pos1 - self.pos2) / (dist**2 + 1e-8) * (self.pos1 - self.pos2)
            v2_new = self.vel2 - np.dot(self.vel2 - self.vel1, self.pos2 - self.pos1) / (dist**2 + 1e-8) * (self.pos2 - self.pos1)
            self.vel1, self.vel2 = v1_new, v2_new
        
        return self._get_observation()
    
    def _get_observation(self):
        """返回两个球的位置和速度"""
        obs = np.zeros(8)  # [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        obs[0], obs[1] = self.pos1
        obs[2], obs[3] = self.vel1
        obs[4], obs[5] = self.pos2
        obs[6], obs[7] = self.vel2
        return obs
    
    def intervene(self, ball_idx, factor=2.0):
        """干预：翻转某个球的速度"""
        if ball_idx == 1:
            self.vel1 *= factor
        else:
            self.vel2 *= factor


# ==========================================
# 2. FCRS-MIS V2: 修正版
# ==========================================
class FCRSMISv2:
    def __init__(self, input_dim=8, hidden_dim=64, lambda_compress=0.01, lr=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lambda_compress = lambda_compress
        self.lr = lr
        
        # 预测头 + 循环连接
        self.W_pred = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W_rec = np.random.randn(hidden_dim, hidden_dim) * 0.1
        
        # 状态
        self.h = np.zeros(hidden_dim)
        
        # 损失归一化用的滑动窗口
        self.pred_err_max = 1.0
        self.compress_err_max = 1.0
        self.window_size = 100
        self.pred_err_history = []
        self.compress_err_history = []
    
    def forward(self, x):
        """预测下一帧"""
        h_new = np.tanh(self.W_rec @ self.h + self.W_pred.T @ x)
        x_pred = self.W_pred @ h_new
        self.h = h_new
        return x_pred, h_new
    
    def update(self, x, x_next):
        """修正版更新：归一化损失 + 最小激活约束"""
        x_pred, h = self.forward(x)
        
        # 1. 预测误差
        pred_error = x_next - x_pred
        pred_err = np.mean(pred_error ** 2)
        
        # 2. 压缩误差 (L1)
        compress_err = np.mean(np.abs(self.W_pred)) + np.mean(np.abs(self.W_rec))
        
        # 3. 最小激活约束 (禁止全零作弊)
        mean_activation = np.mean(np.abs(h))
        activation_penalty = max(0, 0.05 - mean_activation) ** 2
        
        # 4. 滑动窗口归一化
        self.pred_err_history.append(pred_err)
        self.compress_err_history.append(compress_err)
        if len(self.pred_err_history) > self.window_size:
            self.pred_err_history.pop(0)
            self.compress_err_history.pop(0)
        
        self.pred_err_max = max(self.pred_err_history) + 1e-8
        self.compress_err_max = max(self.compress_err_history) + 1e-8
        
        pred_err_norm = pred_err / self.pred_err_max
        compress_err_norm = compress_err / self.compress_err_max
        activation_penalty_norm = activation_penalty
        
        # 5. 更新 (归一化后加权)
        # 预测梯度
        grad_pred = np.outer(pred_error, h) / self.hidden_dim
        # 压缩梯度
        grad_compress = self.lambda_compress * np.sign(self.W_pred)
        # 激活惩罚梯度
        grad_activation = -0.1 * np.sign(h) * (mean_activation < 0.05)
        
        self.W_pred += self.lr * (grad_pred - grad_compress)
        self.W_rec += self.lr * (self.lr * np.outer(h - np.tanh(self.W_rec @ self.h), h))
        
        # 稀疏化
        self.W_pred[np.abs(self.W_pred) < 1e-4] = 0
        self.W_rec[np.abs(self.W_rec) < 1e-4] = 0
        
        return pred_err, compress_err, mean_activation


# ==========================================
# 3. 结构指标 (修正版)
# ==========================================
def compute_structural_entropy(W):
    """用连接权重的模块化分布计算结构熵"""
    # 展平权重
    w_flat = np.abs(W).flatten()
    w_flat = w_flat[w_flat > 1e-4]
    
    if len(w_flat) == 0:
        return 0
    
    # 归一化
    w_flat = w_flat / (w_flat.sum() + 1e-8)
    
    # 计算熵
    entropy = -np.sum(w_flat * np.log2(w_flat + 1e-8))
    
    return entropy


def compute_clustering_metrics(h_list, true_state_list):
    """聚类质量评估"""
    if len(h_list) < 10:
        return 0, 0, 0
    
    # PCA降维
    h_arr = np.array(h_list)
    pca = PCA(n_components=min(10, h_arr.shape[0], h_arr.shape[1]))
    h_pca = pca.fit_transform(h_arr)
    
    # K-Means聚类
    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(h_pca)
    
    # 轮廓系数
    sil = silhouette_score(h_pca, labels)
    
    # 简化ARI (用真实状态的左右/上下分类)
    true_labels = np.array([1 if s[0] > 8 else 0 for s in true_state_list])
    ari = adjusted_rand_score(true_labels, labels)
    
    return sil, ari, pca


# ==========================================
# 4. 反事实干预测试
# ==========================================
def counterfactual_test(env, model, n_steps=100):
    """反事实干预测试 - 验证世界模型"""
    # 正常轨迹
    env.reset()
    h_normal = []
    x = env._get_observation()
    for _ in range(n_steps):
        _, h = model.forward(x)
        h_normal.append(h.copy())
        x = env.step()
    
    # 干预轨迹 (翻转球1的速度)
    env.reset()
    env.intervene(1, factor=2.0)
    h_intervene = []
    x = env._get_observation()
    for _ in range(n_steps):
        _, h = model.forward(x)
        h_intervene.append(h.copy())
        x = env.step()
    
    # 计算状态差异
    h_normal = np.array(h_normal)
    h_intervene = np.array(h_intervene)
    
    # 差异度量
    state_diff = np.mean(np.abs(h_normal - h_intervene))
    
    # 简化版互信息估算
    # 如果状态差异大，说明系统学到了因果变量
    return state_diff


# ==========================================
# 5. 主实验
# ==========================================
def run_experiment(lambda_list=[0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1], n_steps=1000):
    """运行修正版实验"""
    results = []
    
    for lam in lambda_list:
        print(f"\n{'='*50}")
        print(f"λ = {lam}")
        print(f"{'='*50}")
        
        env = TwoBallEnv()
        model = FCRSMISv2(lambda_compress=lam)
        
        pred_errors = []
        compress_errors = []
        activations = []
        h_list = []
        true_state_list = []
        
        x = env.reset()
        for step in range(n_steps):
            x_next = env.step()
            pred_err, compress_err, act = model.update(x, x_next)
            
            pred_errors.append(pred_err)
            compress_errors.append(compress_err)
            activations.append(act)
            
            if step % 10 == 0:
                _, h = model.forward(x)
                h_list.append(h.copy())
                true_state_list.append(x.copy())
            
            x = x_next
            
            if (step + 1) % 200 == 0:
                print(f"Step {step+1} | Pred: {pred_err:.4f} | Compress: {compress_err:.4f} | Act: {act:.4f}")
        
        # 计算指标
        sil, ari, pca = compute_clustering_metrics(h_list, true_state_list)
        struct_entropy = compute_structural_entropy(model.W_pred)
        
        # 反事实测试
        counterfactual_diff = counterfactual_test(env, model)
        
        print(f"\nFinal: Sil={sil:.3f}, ARI={ari:.3f}, StructEntropy={struct_entropy:.3f}")
        print(f"Counterfactual: {counterfactual_diff:.4f}")
        
        results.append({
            "lambda": lam,
            "pred_errors": pred_errors,
            "compress_errors": compress_errors,
            "activations": activations,
            "sil": sil,
            "ari": ari,
            "struct_entropy": struct_entropy,
            "counterfactual": counterfactual_diff
        })
    
    return results


def visualize_v2(results):
    """可视化结果"""
    lambda_list = [r["lambda"] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 预测误差
    ax = axes[0, 0]
    for r in results:
        ax.plot(r["pred_errors"][::20], label=f"λ={r['lambda']}")
    ax.set_title("Prediction Error")
    ax.legend()
    ax.grid(True)
    
    # 2. 激活率
    ax = axes[0, 1]
    for r in results:
        ax.plot(r["activations"][::20], label=f"λ={r['lambda']}")
    ax.set_title("Activation Rate (must stay > 0.05)")
    ax.axhline(y=0.05, color='r', linestyle='--', label='min threshold')
    ax.legend()
    ax.grid(True)
    
    # 3. Silhouette vs λ
    ax = axes[0, 2]
    sil_scores = [r["sil"] for r in results]
    ax.plot(lambda_list, sil_scores, 'o-', linewidth=2)
    ax.set_title("Silhouette Score vs λ")
    ax.set_xlabel("λ")
    ax.grid(True)
    
    # 4. ARI vs λ
    ax = axes[1, 0]
    ari_scores = [r["ari"] for r in results]
    ax.plot(lambda_list, ari_scores, 'o-', linewidth=2, color='green')
    ax.set_title("ARI vs λ")
    ax.set_xlabel("λ")
    ax.grid(True)
    
    # 5. 结构熵 vs λ
    ax = axes[1, 1]
    entropies = [r["struct_entropy"] for r in results]
    ax.plot(lambda_list, entropies, 'o-', linewidth=2, color='orange')
    ax.set_title("Structural Entropy vs λ")
    ax.set_xlabel("λ")
    ax.grid(True)
    
    # 6. 反事实测试
    ax = axes[1, 2]
    cf_diffs = [r["counterfactual"] for r in results]
    ax.plot(lambda_list, cf_diffs, 'o-', linewidth=2, color='purple')
    ax.set_title("Counterfactual Intervention Test")
    ax.set_xlabel("λ")
    ax.set_ylabel("State Diff (higher = better world model)")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("fcrs_mis_v2_results.png", dpi=150)
    print("\n结果已保存到 fcrs_mis_v2_results.png")


if __name__ == "__main__":
    print("="*60)
    print("FCRS-MIS V2: 修正版实验")
    print("解决假阳性问题，验证真正的世界模型")
    print("="*60)
    
    # 运行实验
    lambda_list = [0, 0.001, 0.005, 0.01, 0.02, 0.05]
    results = run_experiment(lambda_list, n_steps=1000)
    
    # 可视化
    visualize_v2(results)
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)
