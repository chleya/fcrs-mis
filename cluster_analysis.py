#!/usr/bin/env python3
"""Analyze what the clusters actually represent"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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


def train_and_analyze(lam):
    """Train model and analyze clusters"""
    W = np.random.randn(64, 6) * 0.1
    h = np.zeros(64)
    
    for _ in range(3000):
        ball = Ball()
        ball.reset()
        traj = [ball.pos.copy()]
        for _ in range(12):
            ball.step()
            traj.append(ball.pos.copy())
        traj = np.array(traj)
        
        x, y = traj[:3].flatten(), traj[-1]
        h = np.tanh(W @ x)
        e = y - (W.T @ h)[:2]
        W += 0.01 * (np.mean(e) * np.mean(h) - lam * np.sign(W))
    
    # Collect data
    h_list, v_list, traj_list = [], [], []
    
    for _ in range(1000):
        ball = Ball()
        ball.reset()
        traj = [ball.pos.copy()]
        for _ in range(12):
            ball.step()
            traj.append(ball.pos.copy())
        traj = np.array(traj)
        
        x = traj[:3].flatten()
        h = np.tanh(W @ x)
        
        h_list.append(h.copy())
        v_list.append(ball.vel.copy())
        traj_list.append(traj.copy())
    
    H = np.array(h_list)
    V = np.array(v_list)
    Traj = np.array(traj_list)
    
    # PCA
    pca = PCA(n_components=10)
    H_pca = pca.fit_transform(H)
    
    # Cluster hidden states
    hidden_clusters = KMeans(n_clusters=3, n_init=10).fit_predict(H_pca)
    
    # Analyze what hidden clusters correspond to
    print(f"\n=== λ = {lam} ===")
    
    # 1. Velocity analysis
    v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
    v_dir = np.arctan2(V[:,1], V[:,0])
    
    print("\n1. Hidden clusters vs Velocity magnitude:")
    for c in range(3):
        mask = hidden_clusters == c
        print(f"   Cluster {c}: v_mag mean = {v_mag[mask].mean():.3f}, std = {v_mag[mask].std():.3f}")
    
    print("\n2. Hidden clusters vs Velocity direction:")
    for c in range(3):
        mask = hidden_clusters == c
        print(f"   Cluster {c}: v_dir mean = {np.degrees(v_dir[mask].mean()):.1f} deg")
    
    # 2. Trajectory shape analysis
    # Calculate trajectory characteristics
    displacements = []
    for traj in Traj:
        d = np.diff(traj, axis=0)
        displacements.append(d)
    Displacements = np.array(displacements)
    
    # Direction of movement
    move_dir = np.arctan2(Displacements[:,:,1].mean(axis=1), Displacements[:,:,0].mean(axis=1))
    
    print("\n3. Hidden clusters vs Trajectory direction:")
    for c in range(3):
        mask = hidden_clusters == c
        print(f"   Cluster {c}: move_dir mean = {np.degrees(move_dir[mask].mean()):.1f} deg")
    
    # 3. Position analysis
    final_pos = Traj[:, -1, :]
    
    print("\n4. Hidden clusters vs Final position:")
    for c in range(3):
        mask = hidden_clusters == c
        print(f"   Cluster {c}: final_x = {final_pos[mask,0].mean():.2f}, final_y = {final_pos[mask,1].mean():.2f}")
    
    # 4. Statistical test: What do clusters best separate?
    from sklearn.metrics import adjusted_rand_score
    
    # Cluster by velocity magnitude bins
    v_bins = np.digitize(v_mag, [0.1, 0.3, 0.5])
    ari_v = adjusted_rand_score(hidden_clusters, v_bins)
    
    # Cluster by velocity direction
    dir_bins = np.digitize(v_dir, [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ari_dir = adjusted_rand_score(hidden_clusters, dir_bins)
    
    # Cluster by trajectory direction
    traj_bins = np.digitize(move_dir, [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ari_traj = adjusted_rand_score(hidden_clusters, traj_bins)
    
    # Cluster by final position quadrant
    quad = (final_pos[:,0] > 8).astype(int) + 2 * (final_pos[:,1] > 8).astype(int)
    ari_pos = adjusted_rand_score(hidden_clusters, quad)
    
    print("\n5. ARI between hidden clusters and different variables:")
    print(f"   vs Velocity magnitude: ARI = {ari_v:.3f}")
    print(f"   vs Velocity direction: ARI = {ari_dir:.3f}")
    print(f"   vs Trajectory direction: ARI = {ari_traj:.3f}")
    print(f"   vs Final position: ARI = {ari_pos:.3f}")
    
    # Find best match
    ari_scores = {
        "velocity_magnitude": ari_v,
        "velocity_direction": ari_dir,
        "trajectory_direction": ari_traj,
        "final_position": ari_pos
    }
    best_match = max(ari_scores, key=ari_scores.get)
    print(f"\n   Best match: {best_match} (ARI = {ari_scores[best_match]:.3f})")
    
    return ari_scores


# Run analysis for different lambdas
print("="*60)
print("CLUSTER MEANING ANALYSIS")
print("="*60)

lams = [0, 0.01, 0.05, 0.1]

all_results = {}
for lam in lams:
    all_results[lam] = train_and_analyze(lam)

# Summary
print("\n" + "="*60)
print("SUMMARY: What do clusters represent?")
print("="*60)

for lam in lams:
    scores = all_results[lam]
    best = max(scores, key=scores.get)
    print(f"λ={lam}: clusters best match {best} (ARI={scores[best]:.3f})")
