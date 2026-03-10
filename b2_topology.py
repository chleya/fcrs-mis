#!/usr/bin/env python3
"""
B2: Topology Invariance Experiment
Verify if phase transition is universal across different network topologies

Topologies: grid, random, small-world, fully-connected
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


def create_topology(n_units, topology_type, seed=42):
    """Create different network topologies"""
    np.random.seed(seed)
    
    if topology_type == 'grid':
        # 2D grid topology
        side = int(np.sqrt(n_units))
        adj = np.zeros((n_units, n_units))
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                # Connect to neighbors
                if i > 0: adj[idx, (i-1)*side+j] = 1
                if i < side-1: adj[idx, (i+1)*side+j] = 1
                if j > 0: adj[idx, i*side+(j-1)] = 1
                if j < side-1: adj[idx, i*side+(j+1)] = 1
        adj = (adj + adj.T) > 0
        return adj.astype(float)
    
    elif topology_type == 'random':
        # Random connections (p=0.1)
        adj = np.random.rand(n_units, n_units) < 0.1
        np.fill_diagonal(adj, 0)
        return adj.astype(float)
    
    elif topology_type == 'small-world':
        # Watts-Strogatz model (p=0.1)
        side = int(np.sqrt(n_units))
        adj = np.zeros((n_units, n_units))
        # Ring lattice
        for i in range(n_units):
            for k in range(1, 3):  # k=1,2 nearest neighbors
                adj[i, (i+k) % n_units] = 1
                adj[i, (i-k) % n_units] = 1
        # Rewire with probability p=0.1
        for i in range(n_units):
            for j in range(i+1, n_units):
                if np.random.rand() < 0.1:
                    new_j = np.random.randint(n_units)
                    adj[i, j] = 0
                    adj[j, i] = 0
                    adj[i, new_j] = 1
                    adj[new_j, i] = 1
        return adj.astype(float)
    
    elif topology_type == 'fully-connected':
        # All-to-all (but we'll use sparse initialization)
        adj = np.ones((n_units, n_units)) - np.eye(n_units)
        return adj.astype(float)
    
    else:
        raise ValueError(f"Unknown topology: {topology_type}")


class Model:
    """Model with configurable topology"""
    def __init__(self, n_hidden=32, lam=0.01, lr=0.01, topology='small-world'):
        self.lam = lam
        self.lr = lr
        self.n_hidden = n_hidden
        
        # Create topology
        adj = create_topology(n_hidden, topology)
        
        # Initialize weights only on connections
        self.W = np.random.randn(n_hidden, 10) * 0.1
        # Zero out non-connections
        self.W = self.W * adj[:, :10]  # Input connections follow topology
        
        self.h = np.zeros(n_hidden)
        self.adj = adj
    
    def forward(self, x):
        self.h = np.tanh(self.W @ x)
        return (self.W.T @ self.h)[:2]
    
    def update(self, x, y):
        p = self.forward(x)
        e = y - p
        mse = np.mean(e**2)
        
        delta = np.mean(e) * np.mean(self.h) - self.lam * np.sign(self.W)
        self.W += self.lr * delta
        self.W[np.abs(self.W) < 1e-4] = 0
        
        return mse


def run_single(topology, lam, seed):
    np.random.seed(seed)
    ball = Ball()
    model = Model(n_hidden=64, lam=lam, topology=topology)
    
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


def run_topology():
    """B2: Topology invariance"""
    topologies = ['grid', 'random', 'small-world', 'fully-connected']
    lams = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    seeds = [42, 123, 456]
    
    results = {}
    
    for topo in topologies:
        print(f"\n{'='*50}")
        print(f"Topology: {topo}")
        print(f"{'='*50}")
        
        results[topo] = []
        
        for lam in lams:
            lam_r = []
            for s in seeds:
                r = run_single(topo, lam, s)
                lam_r.append(r)
            
            avg = {
                "lam": lam,
                "sil": np.mean([x["sil"] for x in lam_r]),
                "mi_v": np.mean([x["mi_v"] for x in lam_r]),
                "sil_std": np.std([x["sil"] for x in lam_r]),
                "mi_v_std": np.std([x["mi_v"] for x in lam_r]),
            }
            results[topo].append(avg)
            print(f"λ={lam:.3f}: Sil={avg['sil']:.3f}, MI(v)={avg['mi_v']:.3f}")
    
    return results


def plot_topology(results):
    """Plot topology invariance results"""
    topologies = list(results.keys())
    lams = [r["lam"] for r in results[topologies[0]]]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Silhouette vs λ for different topologies
    ax = axes[0, 0]
    colors = ['blue', 'red', 'green', 'orange']
    for i, topo in enumerate(topologies):
        sil = [r["sil"] for r in results[topo]]
        ax.plot(lams, sil, 'o-', label=topo, linewidth=2, color=colors[i])
    ax.set_xlabel('λ')
    ax.set_ylabel('Silhouette')
    ax.set_title('Topology Invariance: Silhouette')
    ax.legend()
    ax.grid(True)
    
    # 2. MI(v) vs λ for different topologies
    ax = axes[0, 1]
    for i, topo in enumerate(topologies):
        mi_v = [r["mi_v"] for r in results[topo]]
        ax.plot(lams, mi_v, 'o-', label=topo, linewidth=2, color=colors[i])
    ax.set_xlabel('λ')
    ax.set_ylabel('MI(v)')
    ax.set_title('Topology Invariance: MI(v)')
    ax.legend()
    ax.grid(True)
    
    # 3. Critical λ for each topology
    ax = axes[1, 0]
    critical_lams = []
    for topo in topologies:
        mi_v_list = [r["mi_v"] for r in results[topo]]
        for i, lam in enumerate(lams):
            if mi_v_list[i] > 0.25:
                critical_lams.append(lam)
                break
        else:
            critical_lams.append(lams[-1])
    
    ax.bar(topologies, critical_lams, color=colors)
    ax.set_ylabel('Critical λ')
    ax.set_title('Critical λ by Topology')
    ax.grid(True, axis='y')
    
    # 4. Summary: Phase transition strength
    ax = axes[1, 1]
    phase_strength = []
    for topo in topologies:
        sil_low = results[topo][0]["sil"]  # λ=0
        sil_high = results[topo][-1]["sil"]  # λ=0.1
        phase_strength.append(sil_high - sil_low)
    
    ax.bar(topologies, phase_strength, color=colors)
    ax.set_ylabel('ΔSilhouette (0→0.1)')
    ax.set_title('Phase Transition Strength')
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig("b2_topology_invariance.png", dpi=150)
    print("\nSaved to b2_topology_invariance.png")
    
    # Print summary
    print("\n" + "="*60)
    print("TOPOLOGY INVARIANCE SUMMARY:")
    print("="*60)
    print(f"Topologies tested: {topologies}")
    print(f"Critical λ estimates: {critical_lams}")
    print(f"Phase strength (ΔSil): {phase_strength}")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("B2: Topology Invariance Experiment")
    print("="*60)
    results = run_topology()
    plot_topology(results)
    print("\nDone!")
