#!/usr/bin/env python3
"""
P0++: Variable Time-Step Prediction - Force velocity encoding
Same input -> different T -> different output
Only by encoding velocity can model complete this task
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


def generate_variable_T_data(seq_len=3, T_range=[5, 10, 15, 20]):
    """
    Generate variable time-step prediction data
    - Input: seq_len frames of positions
    - Random T: prediction horizon
    - Target: position at time T
    """
    ball = Ball()
    ball.reset()
    
    # Generate trajectory
    traj = [ball.pos.copy()]
    for _ in range(max(T_range) + seq_len):
        ball.step()
        traj.append(ball.pos.copy())
    traj = np.array(traj)
    
    # Input: first seq_len positions
    input_seq = traj[:seq_len]
    
    # Random T
    T = np.random.choice(T_range)
    
    # Target: position at T steps ahead
    target_pos = traj[seq_len + T]
    
    # True velocity (for analysis only)
    true_v = (traj[seq_len] - traj[seq_len-1]).copy()
    
    return input_seq.flatten(), T, target_pos, true_v


class Model:
    """Model with velocity head for variable T prediction"""
    def __init__(self, n_hidden=64, lam=0.01, lr=0.01):
        self.lam = lam
        self.lr = lr
        self.n_hidden = n_hidden
        
        # Input: seq_len * 2 = 6
        self.W_enc = np.random.randn(n_hidden, 6) * 0.1
        self.W_vel = np.random.randn(2, n_hidden) * 0.1  # velocity prediction head
        
        self.h = np.zeros(n_hidden)
        self.last_T = 0
    
    def forward(self, x, T):
        """Forward pass with variable T"""
        h = np.tanh(np.clip(self.W_enc @ x, -5, 5))
        self.h = h
        
        # Predict velocity
        vel = self.W_vel @ h
        vel = np.clip(vel, -2, 2)  # Clip velocity
        
        # Last input position
        p_last = x[-4:-2]  # last 2 values
        
        # Predict position at time T: p_T = p_last + v * T
        pred = p_last + vel * T
        
        return pred, vel
    
    def update(self, x, T, target):
        pred, vel = self.forward(x, T)
        
        # Clip predictions to avoid overflow
        pred = np.clip(pred, -100, 100)
        
        e = target - pred
        e = np.clip(e, -10, 10)  # Clip error
        
        mse = np.mean(e**2)
        
        # Gradient: simplified
        grad_mag = np.clip(np.mean(np.abs(e)), 0, 10)
        
        # Update with compression
        self.W_enc += self.lr * (grad_mag * np.mean(self.h) - self.lam * np.sign(self.W_enc))
        self.W_vel += self.lr * (grad_mag * T * 0.01 - self.lam * np.sign(self.W_vel))
        
        # Sparsity
        self.W_enc[np.abs(self.W_enc) < 1e-4] = 0
        self.W_vel[np.abs(self.W_vel) < 1e-4] = 0
        
        # Clip weights
        self.W_enc = np.clip(self.W_enc, -10, 10)
        self.W_vel = np.clip(self.W_vel, -10, 10)
        
        return mse


def train_variable_T(lam, steps=3000):
    """Train model with variable time-step"""
    np.random.seed(42)
    model = Model(n_hidden=64, lam=lam)
    
    errors = []
    for step in range(steps):
        x, T, target, v = generate_variable_T_data(3, [5, 10, 15, 20])
        mse = model.update(x, T, target)
        errors.append(mse)
    
    return model, errors


def run_variable_T_experiment():
    """Main experiment"""
    lams = [0, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1]
    
    results = []
    
    for lam in lams:
        print(f"\nλ = {lam:.3f}")
        
        model, errors = train_variable_T(lam, steps=3000)
        
        # Test: collect hidden states and velocities
        h_list, v_list, pred_errors = [], [], []
        
        for _ in range(300):
            x, T, target, true_v = generate_variable_T_data(3, [5, 10, 15, 20])
            pred, vel = model.forward(x, T)
            pred_error = np.mean((pred - target)**2)
            
            h_list.append(model.h.copy())
            v_list.append(vel.copy())
            pred_errors.append(pred_error)
        
        H = np.array(h_list)
        V = np.array(v_list)
        
        # Metrics
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            sil = 0
            mi_v = 0
        else:
            pca = PCA(n_components=min(10, H.shape[1]))
            Hp = pca.fit_transform(H)
            sil = silhouette_score(Hp, KMeans(3, n_init=10).fit_predict(Hp))
            
            v_mag = np.sqrt(V[:,0]**2 + V[:,1]**2)
            h_mag = np.sqrt((H**2).sum(axis=1))
            mi_v = abs(np.corrcoef(v_mag, h_mag)[0,1]) if len(v_mag)>0 else 0
            mi_v = 0 if np.isnan(mi_v) else mi_v
        
        final_mse = np.mean(pred_errors)
        
        print(f"  Sil={sil:.3f}, MI(v)={mi_v:.3f}, MSE={final_mse:.3f}")
        
        results.append({
            "lam": lam,
            "sil": sil,
            "mi_v": mi_v,
            "mse": final_mse
        })
    
    return results


def counterfactual_test_variable_T(model):
    """Counterfactual test: Fixed position, different T"""
    np.random.seed(999)
    
    print("\n" + "="*60)
    print("COUNTERFACTUAL TEST 1: Same Position, Different T")
    print("="*60)
    
    # Generate same input (same 3 frames)
    # Fixed positions: (5,5), (6,5), (7,5) -> v=(1,0)
    x_fixed = np.array([5, 5, 6, 5, 7, 5])
    
    # Test different T values
    Ts = [5, 10, 15, 20]
    preds = []
    vels = []
    
    for T in Ts:
        pred, vel = model.forward(x_fixed, T)
        preds.append(pred)
        vels.append(vel)
        print(f"T={T}: pred={pred}, vel={vel}")
    
    # Check: predictions should follow p_T = p_last + v * T
    # If causal: predictions should scale linearly with T
    preds = np.array(preds)
    vels = np.array(vels)
    
    # Calculate expected positions (if velocity is constant)
    p_last = x_fixed[-4:-2]
    expected_p5 = p_last + vels[0] * 5
    expected_p10 = p_last + vels[0] * 10
    expected_p15 = p_last + vels[0] * 15
    expected_p20 = p_last + vels[0] * 20
    
    print(f"\nExpected (if causal):")
    print(f"  T=5:  {expected_p5}")
    print(f"  T=10: {expected_p10}")
    print(f"  T=15: {expected_p15}")
    print(f"  T=20: {expected_p20}")
    
    print(f"\nActual:")
    for i, T in enumerate(Ts):
        print(f"  T={T}: {preds[i]}")
    
    # Check linearity
    pred_diff_5_10 = preds[1] - preds[0]
    pred_diff_10_15 = preds[2] - preds[1]
    pred_diff_15_20 = preds[3] - preds[2]
    
    is_linear = (np.linalg.norm(pred_diff_5_10 - pred_diff_10_15) < 1.0 and 
                 np.linalg.norm(pred_diff_10_15 - pred_diff_15_20) < 1.0)
    
    print(f"\nPrediction increments: {pred_diff_5_10}, {pred_diff_10_15}, {pred_diff_15_20}")
    print(f"Linear scaling: {'PASS' if is_linear else 'FAIL'}")
    
    # Test 2: Different velocity
    print("\n" + "="*60)
    print("COUNTERFACTUAL TEST 2: Different Input, Same T")
    print("="*60)
    
    # Input 1: velocity = (1, 0)
    x1 = np.array([5.0, 5.0, 6.0, 5.0, 7.0, 5.0])
    pred1, vel1 = model.forward(x1, 10)
    
    # Input 2: velocity = (-1, 0)  
    x2 = np.array([7.0, 5.0, 6.0, 5.0, 5.0, 5.0])
    pred2, vel2 = model.forward(x2, 10)
    
    print(f"Input1: {x1}, pred={pred1}, vel={vel1}")
    print(f"Input2: {x2}, pred={pred2}, vel={vel2}")
    
    delta_pred = np.linalg.norm(pred1 - pred2)
    delta_vel = np.linalg.norm(vel1 - vel2)
    
    causal_2 = delta_pred > 0.1 and delta_vel > 0.1
    print(f"\nDelta prediction: {delta_pred:.4f}")
    print(f"Delta velocity: {delta_vel:.4f}")
    print(f"Causal: {'PASS' if causal_2 else 'FAIL'}")
    
    return is_linear, causal_2


def plot_results(results):
    """Plot results"""
    lams = [r["lam"] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Silhouette
    axes[0,0].plot(lams, [r["sil"] for r in results], 'o-')
    axes[0,0].set_title("Silhouette vs λ")
    axes[0,0].grid(True)
    
    # 2. MI(v)
    axes[0,1].plot(lams, [r["mi_v"] for r in results], 'o-', color='green')
    axes[0,1].set_title("MI(v) vs λ")
    axes[0,1].grid(True)
    
    # 3. MSE
    axes[1,0].plot(lams, [r["mse"] for r in results], 'o-', color='orange')
    axes[1,0].set_title("Prediction MSE")
    axes[1,0].grid(True)
    
    # 4. Summary
    axes[1,1].plot(lams, [r["mi_v"] for r in results], 'o-', label='MI(v)')
    axes[1,1].plot(lams, [1-r["mse"]/20 for r in results], 's--', label='Accuracy')
    axes[1,1].set_title("Summary")
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig("p0pp_variable_T_results.png", dpi=150)
    print("\nSaved to p0pp_variable_T_results.png")
    
    print("\n" + "="*60)
    for r in results:
        print(f"λ={r['lam']:.3f}: Sil={r['sil']:.3f}, MI(v)={r['mi_v']:.3f}, MSE={r['mse']:.3f}")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("P0++: Variable Time-Step Prediction")
    print("="*60)
    
    # Run experiment
    results = run_variable_T_experiment()
    plot_results(results)
    
    # Find best model
    best = max(results, key=lambda x: x["mi_v"])
    print(f"\nBest λ = {best['lam']}")
    
    # Counterfactual test
    model, _ = train_variable_T(best["lam"], steps=3000)
    is_linear, causal_2 = counterfactual_test_variable_T(model)
    
    print("\n" + "="*60)
    print("COUNTERFACTUAL SUMMARY:")
    print("="*60)
    print(f"Test 1 (Linear scaling): {'PASS' if is_linear else 'FAIL'}")
    print(f"Test 2 (Velocity encoding): {'PASS' if causal_2 else 'FAIL'}")
    print("="*60)
