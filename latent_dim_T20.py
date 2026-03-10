#!/usr/bin/env python3
"""Test latent dim with long horizon (T=20)"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

np.random.seed(42)

class Ball:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * 10 + 3
        self.vel = (np.random.rand(2) - 0.5) * 0.4
    
    def step(self):
        self.vel += (np.random.rand(2) - 0.5) * 0.1
        self.vel = np.clip(self.vel, -0.8, 0.8)
        self.pos += self.vel
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)


# Test with different latent dimensions AND T=20
for hidden in [2, 4, 8, 16]:
    lam = 0.01
    T = 20
    
    W = np.random.randn(hidden, 6) * 0.1
    
    for _ in range(3000):
        ball = Ball()
        ball.reset()
        traj = [ball.pos.copy()]
        for _ in range(T + 5):
            ball.step()
            traj.append(ball.pos.copy())
        traj = np.array(traj)
        
        x = traj[:3].flatten()
        y = traj[3 + T]  # predict T steps ahead
        
        h = np.tanh(W @ x)
        e = y - (W.T @ h)[:2]
        W += 0.01 * (np.mean(e) * np.mean(h) - lam * np.sign(W))
    
    # Evaluate
    h_list, v_list = [], []
    for _ in range(500):
        ball = Ball()
        ball.reset()
        traj = [ball.pos.copy()]
        for _ in range(T + 5):
            ball.step()
            traj.append(ball.pos.copy())
        traj = np.array(traj)
        
        x = traj[:3].flatten()
        h = np.tanh(W @ x)
        
        # velocity at prediction time
        v = (traj[3+T] - traj[2+T-1])
        
        h_list.append(h)
        v_list.append(v)
    
    H = np.array(h_list)
    V = np.array(v_list)
    
    v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
    v_bins = np.digitize(v_mag, np.percentile(v_mag, [33, 66]))
    
    if len(set(v_bins)) < 2 or len(set(h_list)) < 2:
        print(f"latent_dim={hidden}, T={T}: Collapsed")
        continue
    
    try:
        h_clusters = KMeans(3, n_init=10).fit_predict(H)
        ari = adjusted_rand_score(h_clusters, v_bins)
        h_mag = np.sqrt((H**2).sum(axis=1))
        mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1])
        mi_v = 0 if np.isnan(mi_v) else mi_v
        print(f"latent_dim={hidden}, T={T}: ARI(v)={ari:.3f}, MI(v)={mi_v:.3f}")
    except:
        print(f"latent_dim={hidden}, T={T}: Error")
