#!/usr/bin/env python3
"""FCRS-MIS V3 Simple"""

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
        return self.pos[0], self.pos[1], self.vel[0], self.vel[1]
    
    def step(self):
        self.pos += self.vel
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)
        return self.pos[0], self.pos[1], self.vel[0], self.vel[1]


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
        
        # Simple update
        self.W += self.lr * (np.mean(e) * np.mean(self.h) - self.lam * np.sign(self.W))
        self.W[np.abs(self.W) < 1e-4] = 0
        
        return mse, comp, np.mean(np.abs(self.h))


def run():
    lams = [0, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02]
    results = []
    
    for lam in lams:
        print(f"\n=== λ = {lam} ===")
        env = BallEnv()
        model = Model(lam=lam)
        
        h_list, s_list = [], []
        env.reset()
        
        for step in range(2000):
            s = np.array(env.reset())
            sn = np.array(env.step())
            mse, comp, act = model.update(s, sn)
            
            if step % 100 == 0:
                h_list.append(model.h.copy())
                s_list.append(s.copy())
            
            if (step+1) % 500 == 0:
                print(f"Step {step+1}: MSE={mse:.4f}")
        
        # Metrics
        H = np.array(h_list)
        S = np.array(s_list)
        
        pca = PCA(n_components=min(10, H.shape[1]))
        Hp = pca.fit_transform(H)
        
        sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
        
        tl = KMeans(3, n_init=10).fit_predict(S[:,:2])
        pl = KMeans(3, n_init=10).fit_predict(Hp)
        ari = adjusted_rand_score(tl, pl)
        
        v = np.sqrt(S[1:,2]**2 + S[1:,3]**2)
        hm = np.sqrt((H[1:]**2).sum(axis=1))
        mi_v = abs(np.corrcoef(v, hm)[0,1]) if len(v)>0 else 0
        
        W = np.abs(model.W.flatten())
        W = W[W>1e-4]
        ent = -np.sum((W/W.sum())*np.log2(W/W.sum())) if len(W)>0 else 0
        
        print(f"Final: Sil={sil:.3f}, ARI={ari:.3f}, MI={mi_v:.3f}")
        results.append({"lam": lam, "sil": sil, "ari": ari, "mi": mi_v, "ent": ent})
    
    return results


def plot(results):
    l = [r["lam"] for r in results]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2,2,1); plt.plot(l, [r["sil"] for r in results], 'o-'); plt.title("Silhouette"); plt.grid(True)
    plt.subplot(2,2,2); plt.plot(l, [r["ari"] for r in results], 'o-', color='green'); plt.title("ARI"); plt.grid(True)
    plt.subplot(2,2,3); plt.plot(l, [r["mi"] for r in results], 'o-', color='purple'); plt.title("MI vs Velocity"); plt.grid(True)
    plt.subplot(2,2,4); plt.plot(l, [r["ent"] for r in results], 'o-', color='orange'); plt.title("Entropy"); plt.grid(True)
    plt.tight_layout()
    plt.savefig("fcrs_mis_v3_results.png", dpi=150)
    print("\nSaved!")


if __name__ == "__main__":
    print("Running V3...")
    r = run()
    plot(r)
    print("Done!")
