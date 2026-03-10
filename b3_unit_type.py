#!/usr/bin/env python3
"""
B3: Unit Type Invariance Experiment
Verify if phase transition is universal across different unit types

Unit types: linear, tanh, relu, sigmoid
"""

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
        self.vel = (np.random.rand(2) - 0.5) * 0.4
        self.history = [self.pos.copy() for _ in range(5)]
        return np.array(self.history).flatten()
    
    def step(self):
        acc = (np.random.rand(2) - 0.5) * 0.1
        self.vel += acc
        self.vel = np.clip(self.vel, -0.8, 0.8)
        self.pos += self.vel
        
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)
        
        self.history.append(self.pos.copy())
        if len(self.history) > 5:
            self.history.pop(0)
        
        return np.array(self.history).flatten()


class Model:
    """Model with configurable activation function"""
    def __init__(self, n_hidden=64, lam=0.01, lr=0.01, activation='tanh'):
        self.lam = lam
        self.lr = lr
        self.activation = activation
        self.n_hidden = n_hidden
        self.W = np.random.randn(n_hidden, 10) * 0.1
        self.h = np.zeros(n_hidden)
    
    def act(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return np.tanh(x)
    
    def forward(self, x):
        h = self.W @ x
        # Clip to prevent overflow
        h = np.clip(h, -10, 10)
        self.h = self.act(h)
        return (self.W.T @ self.h)[:2]
    
    def update(self, x, y):
        p = self.forward(x)
        e = y - p
        mse = np.mean(e**2)
        
        delta = np.mean(e) * np.mean(self.h) - self.lam * np.sign(self.W)
        self.W += self.lr * delta
        self.W[np.abs(self.W) < 1e-4] = 0
        
        return mse


def run_single(activation, lam, seed):
    np.random.seed(seed)
    ball = Ball()
    model = Model(n_hidden=64, lam=lam, activation=activation)
    
    h_list, v_list = [], []
    
    for step in range(3000):
        x = ball.step()
        y = ball.vel.copy()
        model.update(x, y)
        
        if step % 100 == 0:
            h_list.append(model.h.copy())
            v_list.append(ball.vel.copy())
    
    H = np.array(h_list)
    V = np.array(v_list)
    
    # Check for NaN
    if np.any(np.isnan(H)) or np.any(np.isnan(V)):
        return {"sil": 0, "ari": 0, "mi_v": 0}
    
    pca = PCA(n_components=min(10, H.shape[1]))
    Hp = pca.fit_transform(H)
    
    sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
    
    v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
    v_bins = np.digitize(v_mag, np.percentile(v_mag, [33, 66]))
    ari = adjusted_rand_score(v_bins, KMeans(3, n_init=10).fit_predict(Hp))
    
    h_mag = np.sqrt((H**2).sum(axis=1))
    mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1]) if len(v_mag)>0 else 0
    mi_v = 0 if np.isnan(mi_v) else mi_v
    
    return {"sil": sil, "ari": ari, "mi_v": mi_v}


def run_unit_type():
    """B3: Unit type invariance"""
    activations = ['linear', 'tanh', 'relu', 'sigmoid']
    lams = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    seeds = [42, 123, 456]
    
    results = {}
    
    for act in activations:
        print(f"\n{'='*50}")
        print(f"Activation: {act}")
        print(f"{'='*50}")
        
        results[act] = []
        
        for lam in lams:
            lam_r = []
            for s in seeds:
                r = run_single(act, lam, s)
                lam_r.append(r)
            
            avg = {
                "lam": lam,
                "sil": np.mean([x["sil"] for x in lam_r]),
                "mi_v": np.mean([x["mi_v"] for x in lam_r]),
                "sil_std": np.std([x["sil"] for x in lam_r]),
                "mi_v_std": np.std([x["mi_v"] for x in lam_r]),
            }
            results[act].append(avg)
            print(f"λ={lam:.3f}: Sil={avg['sil']:.3f}, MI(v)={avg['mi_v']:.3f}")
    
    return results


def plot_unit_type(results):
    """Plot unit type invariance results"""
    activations = list(results.keys())
    lams = [r["lam"] for r in results[activations[0]]]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. Silhouette vs λ
    ax = axes[0, 0]
    for i, act in enumerate(activations):
        sil = [r["sil"] for r in results[act]]
        ax.plot(lams, sil, 'o-', label=act, linewidth=2, color=colors[i])
    ax.set_xlabel('λ')
    ax.set_ylabel('Silhouette')
    ax.set_title('Unit Type Invariance: Silhouette')
    ax.legend()
    ax.grid(True)
    
    # 2. MI(v) vs λ
    ax = axes[0, 1]
    for i, act in enumerate(activations):
        mi_v = [r["mi_v"] for r in results[act]]
        ax.plot(lams, mi_v, 'o-', label=act, linewidth=2, color=colors[i])
    ax.set_xlabel('λ')
    ax.set_ylabel('MI(v)')
    ax.set_title('Unit Type Invariance: MI(v)')
    ax.legend()
    ax.grid(True)
    
    # 3. Critical λ for each unit type
    ax = axes[1, 0]
    critical_lams = []
    for act in activations:
        mi_v_list = [r["mi_v"] for r in results[act]]
        for i, lam in enumerate(lams):
            if mi_v_list[i] > 0.25:
                critical_lams.append(lam)
                break
        else:
            critical_lams.append(lams[-1])
    
    ax.bar(activations, critical_lams, color=colors)
    ax.set_ylabel('Critical λ')
    ax.set_title('Critical λ by Unit Type')
    ax.grid(True, axis='y')
    
    # 4. Phase transition strength
    ax = axes[1, 1]
    phase_strength = []
    for act in activations:
        sil_low = results[act][0]["sil"]
        sil_high = results[act][-1]["sil"]
        phase_strength.append(sil_high - sil_low)
    
    ax.bar(activations, phase_strength, color=colors)
    ax.set_ylabel('ΔSilhouette (0→0.1)')
    ax.set_title('Phase Transition Strength')
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig("b3_unit_type_invariance.png", dpi=150)
    print("\nSaved to b3_unit_type_invariance.png")
    
    # Print summary
    print("\n" + "="*60)
    print("UNIT TYPE INVARIANCE SUMMARY:")
    print("="*60)
    print(f"Unit types tested: {activations}")
    print(f"Critical λ estimates: {critical_lams}")
    print(f"Phase strength (ΔSil): {phase_strength}")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("B3: Unit Type Invariance Experiment")
    print("="*60)
    results = run_unit_type()
    plot_unit_type(results)
    print("\nDone!")
