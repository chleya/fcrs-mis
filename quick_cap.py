#!/usr/bin/env python3
"""Capacity sweep: dim vs velocity encoding"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

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

def run(dim, T=20, lam=0.01, steps=3000):
    W = np.random.randn(dim, 6) * 0.1
    for _ in range(steps):
        b = Ball()
        t = [b.pos.copy()]
        for _ in range(T+5):
            b.step()
            t.append(b.pos.copy())
        t = np.array(t)
        x, y = t[:3].flatten(), t[-1]
        h = np.tanh(W @ x)
        e = y - (W.T @ h)[:2]
        W += 0.01 * (np.mean(e) * np.mean(h) - lam * np.sign(W))
        W = np.clip(W, -10, 10)
    
    H, V = [], []
    for _ in range(500):
        b = Ball()
        t = [b.pos.copy()]
        for _ in range(T+5):
            b.step()
            t.append(b.pos.copy())
        t = np.array(t)
        H.append(np.tanh(W @ t[:3].flatten()))
        V.append(t[-1] - t[-2])
    
    H, V = np.array(H), np.array(V)
    if np.any(np.isnan(H)) or len(H) < 10:
        return {"dim": dim, "collapsed": True}
    
    sil = silhouette_score(H, KMeans(3, n_init=10).fit_predict(H))
    
    vm = np.sqrt(V[:,0]**2 + V[:,1]**2)
    hm = np.sqrt((H**2).sum(axis=1))
    mi_v = abs(np.corrcoef(vm, hm)[0,1])
    mi_v = 0 if np.isnan(mi_v) else mi_v
    
    return {"dim": dim, "collapsed": False, "sil": sil, "mi_v": mi_v}

print("Capacity Sweep (T=20, lambda=0.01)")
print("="*50)

for d in [64, 32, 16, 8, 4, 2]:
    r = run(d)
    if r.get("collapsed"):
        print(f"dim={d}: COLLAPSED")
    else:
        print(f"dim={d}: Sil={r['sil']:.3f}, MI(v)={r['mi_v']:.3f}")
