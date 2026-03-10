#!/usr/bin/env python3
"""
P0+: Occlusion Prediction - Force velocity encoding
Only by predicting occluded positions can model learn velocity as independent variable
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

np.random.seed(42)

class Ball:
    """Ball with random acceleration"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pos = np.random.rand(2) * 10 + 3
        self.vel = (np.random.rand(2) - 0.5) * 0.4
        self.history = []
        return self
    
    def step(self):
        acc = (np.random.rand(2) - 0.5) * 0.1
        self.vel += acc
        self.vel = np.clip(self.vel, -0.8, 0.8)
        self.pos += self.vel
        
        for i in range(2):
            if self.pos[i] < 1 or self.pos[i] > 14:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 1, 14)
        
        return self.pos.copy()


def generate_occlusion_data(seq_len=3, occlude_steps=10, target_step=13):
    """
    Generate occlusion prediction data
    - Input: seq_len frames of visible positions
    - Target: position at target_step (after occlusion)
    - Model MUST use velocity to predict, cannot use position extrapolation
    """
    ball = Ball()
    ball.reset()
    
    # Generate full trajectory
    full_traj = [ball.pos.copy()]
    for _ in range(target_step - 1):
        ball.step()
        full_traj.append(ball.pos.copy())
    
    full_traj = np.array(full_traj)
    
    # Input: first seq_len positions (visible)
    input_seq = full_traj[:seq_len]
    
    # Target: position at target_step (after occlusion)
    target_pos = full_traj[target_step - 1]
    
    # Also return velocity (for supervision/analysis)
    ball2 = Ball()
    ball2.pos = full_traj[seq_len-1].copy()
    ball2.vel = (full_traj[seq_len] - full_traj[seq_len-1]).copy()
    
    return input_seq.flatten(), target_pos, ball2.vel


class Model:
    """Model for occlusion prediction"""
    def __init__(self, n_hidden=64, lam=0.01, lr=0.01):
        self.lam = lam
        self.lr = lr
        self.n_hidden = n_hidden
        # Input: seq_len * 2 = 3 * 2 = 6
        # Output: 2 (x, y position)
        self.W = np.random.randn(n_hidden, 6) * 0.1
        self.h = np.zeros(n_hidden)
    
    def forward(self, x):
        self.h = np.tanh(self.W @ x)
        # Output: position (2D)
        pred = self.W.T @ self.h
        return pred[:2]
    
    def update(self, x, y):
        p = self.forward(x)
        e = y - p
        mse = np.mean(e**2)
        
        # Simple gradient
        delta = np.mean(e) * np.mean(self.h) - self.lam * np.sign(self.W)
        self.W += self.lr * delta
        self.W[np.abs(self.W) < 1e-4] = 0
        
        return mse


def train_occlusion(lam, steps=3000):
    """Train model with occlusion prediction"""
    np.random.seed(42)
    model = Model(n_hidden=64, lam=lam)
    
    errors = []
    for step in range(steps):
        x, y, v = generate_occlusion_data(3, 10, 13)
        mse = model.update(x, y)
        errors.append(mse)
    
    return model, errors


def run_occlusion_experiment():
    """Main experiment: occlusion prediction"""
    lams = [0, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1]
    
    results = []
    
    for lam in lams:
        print(f"\nλ = {lam:.3f}")
        
        # Train
        model, errors = train_occlusion(lam, steps=3000)
        
        # Test multiple samples
        h_list, v_list, p_list = [], [], []
        
        for _ in range(300):
            x, y, v = generate_occlusion_data(3, 10, 13)
            model.forward(x)
            h_list.append(model.h.copy())
            v_list.append(v.copy())
            p_list.append(y.copy())
        
        H = np.array(h_list)
        V = np.array(v_list)
        P = np.array(p_list)
        
        # Metrics
        pca = PCA(n_components=min(10, H.shape[1]))
        Hp = pca.fit_transform(H)
        sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
        
        v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
        h_mag = np.sqrt((H**2).sum(axis=1))
        mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1]) if len(v_mag)>0 else 0
        mi_v = 0 if np.isnan(mi_v) else mi_v
        
        p_mag = np.sqrt(P[:,0]**2 + P[:,1]**2)
        mi_p = abs(np.corrcoef(p_mag, h_mag)[0,1]) if len(p_mag)>0 else 0
        mi_p = 0 if np.isnan(mi_p) else mi_p
        
        final_mse = np.mean(errors[-500:])
        
        print(f"  Sil={sil:.3f}, MI(v)={mi_v:.3f}, MI(p)={mi_p:.3f}, MSE={final_mse:.3f}")
        
        results.append({
            "lam": lam,
            "sil": sil,
            "mi_v": mi_v,
            "mi_p": mi_p,
            "mse": final_mse
        })
    
    return results


def counterfactual_test_occlusion(model):
    """Counterfactual test with occlusion - THE KEY TEST"""
    np.random.seed(999)
    
    print("\n" + "="*60)
    print("COUNTERFACTUAL TEST (Occlusion)")
    print("="*60)
    
    # Baseline: generate sequence with v=1.0
    x_base, y_base, v_base = generate_occlusion_data(3, 10, 13)
    s_base = model.forward(x_base).copy()
    pred_base = model.W.T @ model.h
    
    print(f"Baseline: v={v_base}, pred={pred_base}")
    
    # Counterfactual: SAME input positions, DIFFERENT velocity
    # Generate target with different velocity, but use SAME input positions
    ball = Ball()
    ball.pos = np.array([8.0, 8.0])  # Fixed position
    
    # Intervention: reverse velocity
    ball.vel = np.array([-1.0, 0.0])  # Left instead of right
    # Calculate what target would be
    for _ in range(12):
        ball.step()
    target_interv = ball.pos.copy()
    
    # Use SAME input as baseline
    x_interv = x_base.copy()
    
    s_interv = model.forward(x_interv).copy()
    pred_interv = model.W.T @ model.h
    
    print(f"Counterfactual: v={ball.vel}, pred={pred_interv}")
    
    # Analysis
    delta_s = np.abs(s_interv - s_base).mean()
    delta_v = np.linalg.norm(ball.vel - v_base)
    delta_pred = np.linalg.norm(pred_interv - pred_base)
    
    print(f"\nDelta internal state: {delta_s:.4f}")
    print(f"Delta velocity: {delta_v:.4f}")
    print(f"Delta prediction: {delta_pred:.4f}")
    
    # Key test: Does changing velocity change the internal state?
    causal = delta_s > 0.01
    
    print(f"\nCAUSAL TEST: {'PASS' if causal else 'FAIL'}")
    print(f"  Model responds to velocity intervention: {causal}")
    
    return {
        "delta_s": delta_s,
        "delta_v": delta_v,
        "delta_pred": delta_pred,
        "causal": causal
    }


def plot_results(results):
    """Plot occlusion experiment results"""
    lams = [r["lam"] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Silhouette
    axes[0,0].plot(lams, [r["sil"] for r in results], 'o-')
    axes[0,0].set_title("Silhouette vs λ")
    axes[0,0].grid(True)
    
    # 2. MI comparison
    axes[0,1].plot(lams, [r["mi_v"] for r in results], 'o-', label='MI(v)')
    axes[0,1].plot(lams, [r["mi_p"] for r in results], 's--', label='MI(p)')
    axes[0,1].axhline(y=0, color='gray', linestyle='--')
    axes[0,1].set_title("MI(v) vs MI(p)")
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # 3. MI difference
    diff = [r["mi_v"] - r["mi_p"] for r in results]
    axes[1,0].plot(lams, diff, 'o-', color='purple')
    axes[1,0].axhline(y=0, color='red', linestyle='--')
    axes[1,0].fill_between(lams, diff, 0, where=[x>0 for x in diff], alpha=0.3, color='green')
    axes[1,0].set_title("MI(v) - MI(p)")
    axes[1,0].grid(True)
    
    # 4. MSE
    axes[1,1].plot(lams, [r["mse"] for r in results], 'o-', color='orange')
    axes[1,1].set_title("Prediction MSE")
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig("p0_occlusion_results.png", dpi=150)
    print("\nSaved to p0_occlusion_results.png")
    
    # Summary
    print("\n" + "="*60)
    print("OCCLUSION EXPERIMENT SUMMARY:")
    print("="*60)
    for r in results:
        status = "OK" if r["mi_v"] > r["mi_p"] else "NO"
        print(f"λ={r['lam']:.3f}: MI(v)={r['mi_v']:.3f}, MI(p)={r['mi_p']:.3f} [{status}]")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("P0+: Occlusion Prediction Experiment")
    print("="*60)
    
    # Run main experiment
    results = run_occlusion_experiment()
    plot_results(results)
    
    # Run counterfactual test on best model
    print("\n" + "="*60)
    print("Finding best model for counterfactual test...")
    print("="*60)
    
    # Find lambda with highest MI(v)
    best_lam = max(results, key=lambda x: x["mi_v"])["lam"]
    print(f"Best λ = {best_lam}")
    
    model, _ = train_occlusion(best_lam, steps=3000)
    cf_result = counterfactual_test_occlusion(model)
    
    print("\nDone!")
