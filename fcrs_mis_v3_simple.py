#!/usr/bin/env python3
"""
FCRS-MIS V3: Simplified

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
# 1. Environment
# ==========================================
class SingleBallEnv:
    def __init__(self, grid_size=16, speed=0.3):
        self.grid_size = grid_size
        self.speed = speed
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * 10 + 3
        self.vel = (np.random.rand(2) - 0.5) * speed
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
    
    def get_next_state(self):
        """Just predict 1 step"""
        return self.step()


speed = 0.3

# ==========================================
# 2. Simple Model
# ==========================================
class SimpleModel:
    def __init__(self, state_dim=4, hidden_dim=32, lambda_compress=0.01, lr=0.01):
        self.lambda_c = lambda_compress
        self.lr = lr
        
        # Simple: state -> hidden -> next_state
        self.W = np.random.randn(hidden_dim, state_dim) * 0.1
        self.h = np.zeros(hidden_dim)
        
        self.hist = []
    
    def forward(self, state):
        self.h = np.tanh(self.W @ state)
        pred = self.W.T @ self.h
        return pred
    
    def update(self, state, next_state):
        pred = self.forward(state)
        err = next_state - pred
        
        # MSE loss
        mse = np.mean(err ** 2)
        
        # Compression loss
        compress = np.mean(np.abs(self.W))
        
        # Update - simple Hebbian-like
        delta = np.outer(err, self.h) - self.lambda_c * np.sign(self.W)
        delta = np.mean(err) * np.mean(self.h) - self.lambda_c * np.sign(self.W)
        
        self.W += self.lr * delta
        
        # Sparse
        self.W[np.abs(self.W) < 1e-4] = 0
        
        return mse, compress, np.mean(np.abs(self.h))


# ==========================================
# 3. Metrics
# ==========================================
def metrics(h_list, state_list):
    if len(h_list) < 10:
        return 0, 0, 0, 0
    
    H = np.array(h_list)
    S = np.array(state_list)
    
    # PCA
    pca = PCA(n_components=min(10, H.shape[1]))
    H_pca = pca.fit_transform(H)
    
    # Silhouette
    sil = silhouette_score(H_pca, KMeans(n_clusters=3, n_init=10).fit_predict(H_pca))
    
    # ARI
    true_lab = KMeans(n_clusters=3, n_init=10).fit_predict(S[:, :2])
    pred_lab = KMeans(n_clusters=3, n_init=10).fit_predict(H_pca)
    ari = adjusted_rand_score(true_lab, pred_lab)
    
    # MI
    vel = np.sqrt(S[1:, 2]**2 + S[1:, 3]**2)
    h_mag = np.sqrt((H[1:]**2).sum(axis=1))
    pos = np.sqrt(S[1:, 0]**2 + S[1:, 1]**2)
    
    mi_vel = abs(np.corrcoef(vel, h_mag)[0,1]) if len(vel)>0 else 0
    mi_pos = abs(np.corrcoef(pos, h_mag)[0,1]) if len(pos)>0 else 0
    
    # Entropy
    W = np.abs(model.W.flatten())
    W = W[W > 1e-4]
    ent = -np.sum((W/W.sum()) * np.log2(W/W.sum() + 1e-8)) if len(W)>0 else 0
    
    return sil, ari, mi_vel, mi_pos, ent


# ==========================================
# 4. Main
# ==========================================
def run():
    lambdas = [0, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02]
    results = []
    
    for lam in lambdas:
        print(f"\n=== λ = {lam} ===")
        
        env = SingleBallEnv()
        model = SimpleModel(lambda_compress=lam)
        
        h_list, state_list = [], []
        
        env.reset()
        for step in range(2000):
            s = env._get_state()
            ns = env.get_next_state()
            mse, comp, act = model.update(s, ns)
            
            if step % 100 == 0:
                h_list.append(model.h.copy())
                state_list.append(s.copy())
            
            if (step+1) % 500 == 0:
                print(f"Step {step+1}: MSE={mse:.4f}, Comp={comp:.4f}")
        
        sil, ari, mi_vel, mi_pos, ent = metrics(h_list, state_list)
        print(f"Final: Sil={sil:.3f}, ARI={ari:.3f}, MI_vel={mi_vel:.3f}, MI_pos={mi_pos:.3f}")
        
        results.append({"lambda": lam, "sil": sil, "ari": ari, "mi_vel": mi_vel, "mi_pos": mi_pos, "ent": ent})
    
    return results


def plot(results):
    lams = [r["lambda"] for r in results]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(lams, [r["sil"] for r in results], 'o-')
    plt.title("Silhouette")
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(lams, [r["ari"] for r in results], 'o-', color='green')
    plt.title("ARI")
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(lams, [r["mi_vel"] for r in results], 'o-', label='vel')
    plt.plot(lams, [r["mi_pos"] for r in results], 's--', label='pos')
    plt.title("MI")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(lams, [r["ent"] for r in results], 'o-', color='orange')
    plt.title("Entropy")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("fcrs_mis_v3_results.png", dpi=150)
    print("\nSaved!")


if __name__ == "__main__":
    print("FCRS-MIS V3 Running...")
    results = run()
    plot(results)
    print("\nDone!")
