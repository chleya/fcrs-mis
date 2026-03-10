#!/usr/bin/env python3
"""FCRS-MIS V5: Long-term Prediction"""

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
        # predict position only (2D)
        pred_pos = self.W.T @ self.h
        return pred_pos[:2]  # only position
    
    def update(self, x, y):
        p = self.forward(x)
        e = y - p
        mse = np.mean(e**2)
        comp = np.mean(np.abs(self.W))
        delta = np.mean(e) * np.mean(self.h) - self.lam * np.sign(self.W)
        self.W += self.lr * delta
        self.W[np.abs(self.W) < 1e-4] = 0
        return mse, comp


def run():
    lams = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    results = []
    
    for lam in lams:
        print(f"\n=== λ = {lam} ===")
        ball = Ball()
        model = Model(lam=lam)
        
        h_list, v_list, p_list = [], [], []
        
        for step in range(3000):
            x = ball.reset()
            # predict next 10 steps
            y = ball.pos.flatten()  # just predict next pos
            
            mse, comp = model.update(x, y)
            
            if step % 100 == 0:
                h_list.append(model.h.copy())
                v_list.append(ball.vel.copy())
                p_list.append(ball.pos.copy())
            
            if (step+1) % 1000 == 0:
                print(f"Step {step+1}: MSE={mse:.4f}")
        
        H = np.array(h_list)
        V = np.array(v_list)
        P = np.array(p_list)
        
        pca = PCA(n_components=min(10, H.shape[1]))
        Hp = pca.fit_transform(H)
        sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
        
        v_dir = np.arctan2(V[:,1], V[:,0])
        v_bins = np.digitize(v_dir, [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        true_lab = v_bins
        pred_lab = KMeans(3, n_init=10).fit_predict(Hp)
        ari = adjusted_rand_score(true_lab, pred_lab)
        
        v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
        h_mag = np.sqrt((H**2).sum(axis=1))
        mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1]) if len(v_mag)>0 else 0
        
        p_mag = np.sqrt(P[:,0]**2 + P[:,1]**2)
        mi_p = abs(np.corrcoef(p_mag, h_mag)[0,1]) if len(p_mag)>0 else 0
        
        W = np.abs(model.W.flatten())
        W = W[W>1e-4]
        ent = -np.sum((W/W.sum())*np.log2(W/W.sum())) if len(W)>0 else 0
        
        print(f"Final: Sil={sil:.3f}, ARI={ari:.3f}, MI(v)={mi_v:.3f}, MI(p)={mi_p:.3f}")
        results.append({"lam": lam, "sil": sil, "ari": ari, "mi_v": mi_v, "mi_p": mi_p, "ent": ent})
    
    return results


def plot(results):
    lams = [r["lam"] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2,3,1)
    plt.plot(lams, [r["sil"] for r in results], 'o-')
    plt.title("Silhouette"); plt.grid(True)
    
    plt.subplot(2,3,2)
    plt.plot(lams, [r["ari"] for r in results], 'o-', color='green')
    plt.title("ARI"); plt.grid(True)
    
    plt.subplot(2,3,3)
    plt.plot(lams, [r["mi_v"] for r in results], 'o-', label='MI(v)')
    plt.plot(lams, [r["mi_p"] for r in results], 's--', label='MI(p)')
    plt.title("MI Comparison"); plt.legend(); plt.grid(True)
    
    plt.subplot(2,3,4)
    plt.plot(lams, [r["ent"] for r in results], 'o-', color='orange')
    plt.title("Entropy"); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("fcrs_mis_v5.png", dpi=150)
    print("\nSaved!")
    
    print("\n" + "="*50)
    for r in results:
        status = "OK" if r["mi_v"] > r["mi_p"] else "NO"
        print(f"λ={r['lam']}: MI(v)={r['mi_v']:.3f}, MI(p)={r['mi_p']:.3f} [{status}]")
    print("="*50)


if __name__ == "__main__":
    print("V5 Running...")
    r = run()
    plot(r)
    print("Done!")
