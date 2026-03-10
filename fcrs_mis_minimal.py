#!/usr/bin/env python3
"""
FCRS-MIS: FCRS Minimal Experiment System
验证「压缩约束是否导致结构形成」

Author: NeuralSite Team
Date: 2026-03-10
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

# ==========================================
# 1. 极简环境：MovingDotEnv (Level 0)
# 功能：生成单个点在二维平面随机移动的序列
# ==========================================
class MovingDotEnv:
    def __init__(self, grid_size=16, speed=0.1):
        self.grid_size = grid_size
        self.speed = speed
        self.reset()
    
    def reset(self):
        # 随机初始化位置和速度
        self.pos = np.random.rand(2) * (self.grid_size - 2) + 1
        self.vel = (np.random.rand(2) - 0.5) * 2 * self.speed
        return self._get_observation()
    
    def step(self):
        # 更新位置，边界反弹
        self.pos += self.vel
        for i in range(2):
            if self.pos[i] <= 0 or self.pos[i] >= self.grid_size - 1:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, self.grid_size - 2)
        return self._get_observation()
    
    def _get_observation(self):
        # 将位置转换为二维网格观测 (one-hot-like)
        obs = np.zeros((self.grid_size, self.grid_size))
        x, y = int(np.round(self.pos[0])), int(np.round(self.pos[1]))
        obs[x, y] = 1.0
        return obs.flatten()


# ==========================================
# 2. FCRS-MIS 微单元系统 + 预测编码更新
# 功能：100单元局部连接，预测编码局部误差传播
# ==========================================
class FCRSMIS:
    def __init__(self, input_dim=256, n_units=100, lambda_compress=0.1, lr=0.01):
        self.input_dim = input_dim
        self.n_units = n_units
        self.lambda_compress = lambda_compress  # 压缩权重λ
        self.lr = lr
        
        # 初始化：局部连接拓扑 (小世界-like，每个单元只连输入局部区域和相邻单元)
        self.W_in = np.zeros((n_units, input_dim))  # 输入连接
        self.W_rec = np.zeros((n_units, n_units))  # 循环连接
        
        # 每个单元只关注输入的一个小局部区域
        grid_size = int(np.sqrt(input_dim))
        receptive_field_size = 5
        for i in range(n_units):
            # 随机选择局部感受野中心
            cx, cy = np.random.randint(receptive_field_size, grid_size - receptive_field_size, 2)
            # 初始化局部输入连接
            for dx in range(-receptive_field_size, receptive_field_size+1):
                for dy in range(-receptive_field_size, receptive_field_size+1):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < grid_size and 0 <= y < grid_size:
                        idx = x * grid_size + y
                        self.W_in[i, idx] = np.random.randn() * 0.1
        
        # 稀疏循环连接 (每个单元只连最近的5个邻居)
        for i in range(n_units):
            neighbors = np.random.choice(n_units, 5, replace=False)
            self.W_rec[i, neighbors] = np.random.randn() * 0.1
        
        # 状态
        self.s = np.zeros(n_units)  # 内部状态
        self.s_hat = np.zeros(n_units)  # 预测状态
    
    def forward(self, x):
        """前向传播：预测 + 激活"""
        # 预测当前状态 (基于上一时刻)
        self.s_hat = np.tanh(self.W_rec @ self.s)
        # 更新当前状态 (基于输入 + 预测)
        self.s = np.tanh(self.W_in @ x + self.W_rec @ self.s)
        return self.s, self.s_hat
    
    def update(self, x, x_next):
        """预测编码型局部更新：仅使用局部预测误差"""
        s, s_hat = self.forward(x)
        
        # 1. 计算预测误差 (对下一时刻输入的预测误差)
        # 简单线性预测头：从状态预测下一时刻输入
        x_pred = self.W_in.T @ s  # 用输入连接的转置作为预测头
        pred_error = x_next - x_pred
        
        # 2. 局部更新输入连接 (Hebbian + 误差调制)
        delta_W_in = np.outer(pred_error, s).T - self.lambda_compress * np.sign(self.W_in)
        self.W_in += self.lr * delta_W_in
        
        # 3. 局部更新循环连接 (基于状态预测误差)
        state_pred_error = s - s_hat
        delta_W_rec = np.outer(state_pred_error, self.s).T - self.lambda_compress * np.sign(self.W_rec)
        self.W_rec += self.lr * delta_W_rec
        
        # 4. 稀疏化：裁剪极小权重
        self.W_in[np.abs(self.W_in) < 1e-4] = 0
        self.W_rec[np.abs(self.W_rec) < 1e-4] = 0
        
        return np.mean(pred_error**2), np.mean(np.abs(s))  # 返回预测误差和稀疏度


# ==========================================
# 3. 结构涌现指标计算
# ==========================================
def compute_metrics(s_list, true_pos_list):
    """
    计算结构指标：
    - 轮廓系数 (聚类质量)
    - 结构熵 (有序程度)
    - ARI (如果有真实标签)
    """
    # 1. 聚类指标
    if len(s_list) < 10:
        return 0, 0, 0
    
    # 用PCA降维
    pca = PCA(n_components=10)
    s_pca = pca.fit_transform(s_list)
    
    # K-Means聚类 (假设聚2类：点的位置左右/上下)
    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(s_pca)
    
    # 轮廓系数
    sil_score = silhouette_score(s_pca, labels)
    
    # 2. 结构熵 (基于连接权重的分布)
    # 这里简化为状态激活的熵
    s_flat = np.concatenate(s_list)
    s_flat = (s_flat - s_flat.min()) / (s_flat.max() - s_flat.min() + 1e-8)
    hist, _ = np.histogram(s_flat, bins=20, density=True)
    hist = hist[hist > 0]
    struct_entropy = -np.sum(hist * np.log2(hist))
    
    # 3. 简化ARI：用真实位置的左右作为真实标签
    true_labels = (np.array(true_pos_list)[:, 0] > 8).astype(int)
    ari_score = adjusted_rand_score(true_labels, labels)
    
    return sil_score, struct_entropy, ari_score


# ==========================================
# 4. 单组实验运行
# ==========================================
def run_single_experiment(lambda_compress, n_steps=2000, log_interval=200):
    """运行一组实验，返回指标曲线"""
    env = MovingDotEnv()
    model = FCRSMIS(lambda_compress=lambda_compress)
    
    pred_errors = []
    sparsities = []
    s_list = []
    true_pos_list = []
    
    x = env.reset()
    for step in range(n_steps):
        x_next = env.step()
        pred_err, sparsity = model.update(x, x_next)
        
        # 记录
        pred_errors.append(pred_err)
        sparsities.append(sparsity)
        if step % 10 == 0:
            s_list.append(model.s.copy())
            true_pos_list.append(env.pos.copy())
        
        x = x_next
        
        # 日志
        if (step + 1) % log_interval == 0:
            print(f"λ={lambda_compress:.3f} | Step {step+1}/{n_steps} | Pred Err: {pred_err:.4f} | Sparsity: {sparsity:.4f}")
    
    # 计算最终结构指标
    sil_score, struct_entropy, ari_score = compute_metrics(s_list, true_pos_list)
    print(f"\nλ={lambda_compress:.3f} | Final: Sil={sil_score:.3f}, Entropy={struct_entropy:.3f}, ARI={ari_score:.3f}\n")
    
    return {
        "lambda": lambda_compress,
        "pred_errors": np.array(pred_errors),
        "sparsities": np.array(sparsities),
        "sil_score": sil_score,
        "struct_entropy": struct_entropy,
        "ari_score": ari_score,
        "final_s_list": s_list[-100:],
        "final_pos_list": true_pos_list[-100:]
    }


# ==========================================
# 5. 相变实验：遍历λ，观察结构涌现
# ==========================================
def run_phase_transition_experiment(lambda_list=[0, 0.01, 0.1, 0.5, 1.0]):
    """运行相变实验，绘制临界曲线"""
    results = []
    for lam in lambda_list:
        res = run_single_experiment(lam)
        results.append(res)
    return results


# ==========================================
# 6. 可视化工具
# ==========================================
def visualize_results(results):
    """可视化所有结果：预测误差、稀疏度、相变曲线、聚类"""
    lambda_list = [r["lambda"] for r in results]
    sil_scores = [r["sil_score"] for r in results]
    struct_entropies = [r["struct_entropy"] for r in results]
    ari_scores = [r["ari_score"] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 预测误差曲线 (选λ=0和λ=0.1对比)
    ax = axes[0, 0]
    for r in results:
        if r["lambda"] in [0, 0.1]:
            ax.plot(r["pred_errors"][::20], label=f"λ={r['lambda']}")
    ax.set_title("Prediction Error (smoothed)")
    ax.set_xlabel("Step (x20)")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True)
    
    # 2. 稀疏度曲线
    ax = axes[0, 1]
    for r in results:
        if r["lambda"] in [0, 0.1]:
            ax.plot(r["sparsities"][::20], label=f"λ={r['lambda']}")
    ax.set_title("Activation Sparsity (smoothed)")
    ax.set_xlabel("Step (x20)")
    ax.set_ylabel("L1 Norm")
    ax.legend()
    ax.grid(True)
    
    # 3. 相变曲线：Silhouette Score vs λ
    ax = axes[0, 2]
    ax.plot(lambda_list, sil_scores, 'o-', linewidth=2, markersize=8)
    ax.set_title("Phase Transition: Clustering Quality vs λ")
    ax.set_xlabel("Compression Weight λ")
    ax.set_ylabel("Silhouette Score")
    ax.grid(True)
    
    # 4. 相变曲线：Structure Entropy vs λ
    ax = axes[1, 0]
    ax.plot(lambda_list, struct_entropies, 'o-', linewidth=2, markersize=8, color='orange')
    ax.set_title("Phase Transition: Structure Entropy vs λ")
    ax.set_xlabel("Compression Weight λ")
    ax.set_ylabel("Structure Entropy (lower = more ordered)")
    ax.grid(True)
    
    # 5. 相变曲线：ARI vs λ
    ax = axes[1, 1]
    ax.plot(lambda_list, ari_scores, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_title("Phase Transition: ARI vs λ")
    ax.set_xlabel("Compression Weight λ")
    ax.set_ylabel("Adjusted Rand Index")
    ax.grid(True)
    
    # 6. 可视化最佳λ的内部状态聚类
    best_idx = np.argmax(sil_scores)
    best_res = results[best_idx]
    ax = axes[1, 2]
    
    # PCA降维可视化
    s_pca = PCA(n_components=2).fit_transform(best_res["final_s_list"])
    pos = np.array(best_res["final_pos_list"])
    colors = (pos[:, 0] > 8).astype(int)  # 按左右位置着色
    
    scatter = ax.scatter(s_pca[:, 0], s_pca[:, 1], c=colors, cmap='coolwarm', alpha=0.7)
    ax.set_title(f"Internal State Clustering (Best λ={best_res['lambda']:.3f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="True Position (Left/Right)")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("fcrs_mis_results.png", dpi=150)
    print("可视化结果已保存至 fcrs_mis_results.png")
    plt.show()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("FCRS-MIS 最小实验系统启动")
    print("目标：验证压缩约束λ是否导致结构涌现")
    print("="*60)
    
    # 1. 运行相变实验
    lambda_list = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = run_phase_transition_experiment(lambda_list)
    
    # 2. 可视化结果
    visualize_results(results)
    
    print("\n" + "="*60)
    print("实验完成！")
    print("请查看 fcrs_mis_results.png 分析结构涌现的相变现象。")
    print("="*60)
