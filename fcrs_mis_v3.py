#!/usr/bin/env python3
"""
FCRS-MIS V3: 极简验证版

Author: NeuralSite Team
Date: 2026-03-10
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

np.random.seed(42)

# ==========================================
# 1. 单球环境
# ==========================================
class SingleBallEnv:
    def __init__(self, grid_size=16, speed=0.3):
        self.grid_size = grid_size
        self.speed = speed
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * (self.grid_size - 4) + 2
        self.vel = (np.random.rand(2) - 0.5) * 2 * self.speed
        return self._get_state()
    
    def step(self):
        self.pos += self.vel
        for i in range(2):
            if self.pos[i] <= 1 or self.pos[i] >= self.grid_size - 1:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, self.grid_size - 1)
        return self._get_state()
    
    def _get_state(self):
        return np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1]])
    
    def get_trajectory(self, steps=10):
        traj = []
        state = self._get_state()
        for _ in range(steps):
            traj.append(state)
            self.step()
            state = self._get_state()
        return np.array(traj)


# ==========================================
# 2. FCRS-MIS V3
# ==========================================
class FCRSMISv3:
    def __init__(self, state_dim=4, hidden_dim=32, predict_steps=10, lambda_compress=0.01, lr=0.01):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.predict_steps = predict_steps
        self.lambda_compress = lambda_compress
        self.lr = lr
        
        self.W_enc = np.random.randn(hidden_dim, state_dim) * 0.1
        self.W_rec = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_dec = np.random.randn(predict_steps * 2, hidden_dim) * 0.1
        
        self.h = np.zeros(hidden_dim)
        self.history = []
    
    def forward(self, state):
        self.h = np.tanh(self.W_enc @ state + self.W_rec @ self.h)
        traj_pred = self.W_dec @ self.h
        return traj_pred.reshape(self.predict_steps, 2)
    
    def update(self, state, true_traj, step, total_steps):
        traj_pred = self.forward(state)
        pred_error = true_traj[:, :2] - traj_pred
        pred_loss = np.mean(pred_error ** 2)
        
        compress_loss = (np.mean(np.abs(self.W_enc)) + 
                       np.mean(np.abs(self.W_rec)) + 
                       np.mean(np.abs(self.W_dec))) / 3
        
        activation = np.mean(np.abs(self.h))
        
        lambda_eff = self.lambda_compress * (0.5 if step < 500 else 1.0)
        
        # 简化梯度
        error_grad = -pred_error.T @ self.W_dec / (self.predict_steps * 2)
        
        grad_enc = np.outer(error_grad.mean(), state) - lambda_eff * np.sign(self.W_enc)
        grad_rec = np.outer(error_grad.mean(), self.h) - lambda_eff * np.sign(self.W_rec)
        grad_dec = np.outer(pred_error.mean(axis=0), self.h) - lambda_eff * np.sign(self.W_dec)
        
        self.W_enc += self.lr * grad_enc
        self.W_rec += self.lr * grad_rec
        self.W_dec += self.lr * grad_dec
        
        self.W_enc[np.abs(self.W_enc) < 1e-4] = 0
        self.W_rec[np.abs(self.W_rec) < 1e-4] = 0
        self.W_dec[np.abs(self.W_dec) < 1e-4] = 0
        
        return pred_loss, compress_loss, activation


# ==========================================
# 3. 评估
# ==========================================
def compute_ari(h_list, state_list):
    if len(h_list) < 10:
        return 0
    h_arr = np.array(h_list)
    pca = PCA(n_components=min(10, h_arr.shape[1]))
    h_pca = pca.fit_transform(h_arr)
    pred_labels = KMeans(n_clusters=3, n_init=10).fit_predict(h_pca)
    states = np.array(state_list)
    true_labels = KMeans(n_clusters=3, n_init=10).fit_predict(states[:, :2])
    return adjusted_rand_score(true_labels, pred_labels)


def compute_mi(h_list, state_list):
    if len(h_list) < 10:
        return 0, 0
    h_arr = np.array(h_list)
    states = np.array(state_list)
    vel_mag = np.sqrt(states[1:, 2]**2 + states[1:, 3]**2)
    h_mag = np.sqrt((h_arr[1:]**2).sum(axis=1))
    pos_mag = np.sqrt(states[1:, 0]**2 + states[1:, 1]**2)
    
    mi_vel = abs(np.corrcoef(vel_mag, h_mag)[0, 1]) if len(vel_mag) > 0 else 0
    mi_pos = abs(np.corrcoef(pos_mag, h_mag)[0, 1]) if len(pos_mag) > 0 else 0
    return (mi_vel if not np.isnan(mi_vel) else 0, 
            mi_pos if not np.isnan(mi_pos) else 0)


# ==========================================
# 4. 主实验
# ==========================================
def run_v3():
    lambda_list = [0, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02]
    results = []
    
    for lam in lambda_list:
        print(f"\n{'='*40}\nλ = {lam}\n{'='*40}")
        
        env = SingleBallEnv()
        model = FCRSMISv3(lambda_compress=lam, hidden_dim=32)
        
        h_list, state_list = [], []
        
        env.reset()
        for step in range(2000):
            state = env._get_state()
            traj = env.get_trajectory(10)
            model.update(state, traj, step, 2000)
            
            if step % 100 == 0:
                h_list.append(model.h.copy())
                state_list.append(state.copy())
            
            if (step + 1) % 500 == 0:
                print(f"Step {step+1}")
        
        ari = compute_ari(h_list, state_list)
        mi_vel, mi_obs = compute_mi(h_list, state_list)
        sil = silhouette_score(np.array(h_list), KMeans(n_clusters=3, n_init=10).fit_predict(np.array(h_list)))
        
        W_all = np.concatenate([model.W_enc.flatten(), model.W_rec.flatten(), model.W_dec.flatten()])
        W_nz = W_all[np.abs(W_all) > 1e-4]
        if len(W_nz) > 0:
            W_norm = np.abs(W_nz) / (np.abs(W_nz).sum() + 1e-8)
            struct_ent = -np.sum(W_norm * np.log2(W_norm + 1e-8))
        else:
            struct_ent = 0
        
        print(f"Final: Sil={sil:.3f}, ARI={ari:.3f}, MI_vel={mi_vel:.3f}, MI_obs={mi_obs:.3f}")
        
        results.append({"lambda": lam, "sil": sil, "ari": ari, "mi_vel": mi_vel, "mi_obs": mi_obs, "struct_ent": struct_ent})
    
    return results


def visualize(results):
    lambda_list = [r["lambda"] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].plot(lambda_list, [r["sil"] for r in results], 'o-')
    axes[0,0].set_title("Silhouette vs λ")
    axes[0,0].grid(True)
    
    axes[0,1].plot(lambda_list, [r["ari"] for r in results], 'o-', color='green')
    axes[0,1].set_title("ARI vs λ")
    axes[0,1].grid(True)
    
    axes[0,2].plot(lambda_list, [r["mi_vel"] for r in results], 'o-', label='vs velocity')
    axes[0,2].plot(lambda_list, [r["mi_obs"] for r in results], 's--', label='vs position')
    axes[0,2].set_title("Mutual Information")
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    axes[1,0].plot(lambda_list, [r["struct_ent"] for r in results], 'o-', color='orange')
    axes[1,0].set_title("Structural Entropy vs λ")
    axes[1,0].grid(True)
    
    plt.tight_layout()
    plt.savefig("fcrs_mis_v3_results.png", dpi=150)
    print("\nSaved to fcrs_mis_v3_results.png")


if __name__ == "__main__":
    print("FCRS-MIS V3 Running...")
    results = run_v3()
    visualize(results)
    print("\nDone!")
