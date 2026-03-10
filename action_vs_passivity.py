#!/usr/bin/env python3
"""
Action vs Passivity: Causal State Abstraction Experiment
=========================================================
Strict control: only difference is "action permission"
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)


class ActableMovingPointEnv:
    """Environment with action intervention capability"""
    
    def __init__(self, alpha=1.0, max_action=0.5):
        self.alpha = alpha
        self.max_action = max_action
    
    def step(self, x, y, vx, vy, action):
        """Action directly干预 velocity, not position"""
        new_vx = vx + action[0]
        new_vy = vy + action[1]
        new_vx = np.clip(new_vx, -2, 2)
        new_vy = np.clip(new_vy, -2, 2)
        
        new_x = x + self.alpha * new_vx
        new_y = y + self.alpha * new_vy
        
        return new_x, new_y, new_vx, new_vy
    
    def generate_passive_sequence(self, seq_len=3, pred_len=10):
        """Passive: fixed velocity random trajectory"""
        vx, vy = np.random.uniform(-1, 1, 2)
        x, y = 0, 0
        obs = []
        for _ in range(seq_len + pred_len):
            obs.append([x, y])
            x += self.alpha * vx
            y += self.alpha * vy
        return np.array(obs[:seq_len]), np.array(obs[seq_len:]), vx, vy
    
    def generate_active_sequence(self, model, seq_len=3, pred_len=10):
        """Active: model outputs actions to intervene"""
        vx, vy = np.random.uniform(-1, 1, 2)
        x, y = 0, 0
        obs = []
        
        # Initial frames without action
        for _ in range(seq_len):
            obs.append([x, y])
            x += self.alpha * vx
            y += self.alpha * vy
        
        # Subsequent frames with model action
        for _ in range(pred_len):
            current_obs = np.array(obs[-seq_len:]).flatten()
            action = model.predict_action(current_obs)
            x, y, vx, vy = self.step(x, y, vx, vy, action)
            obs.append([x, y])
        
        return np.array(obs[:seq_len]), np.array(obs[seq_len:]), vx, vy


class PredictiveModel:
    """Model with optional action output"""
    
    def __init__(self, n_hidden=32, n_action=2, has_action=False, lr=0.01, lam=0.01):
        self.n_hidden = n_hidden
        self.has_action = has_action
        self.lr = lr
        self.lam = lam
        
        # Weights: input→hidden, hidden→position prediction
        self.W1 = np.random.randn(n_hidden, 6) * 0.1
        self.h = np.zeros(n_hidden)
        
        if has_action:
            # Additional head for action prediction
            self.W_action = np.random.randn(n_action, n_hidden) * 0.1
    
    def forward(self, x):
        self.h = np.tanh(self.W1 @ x)
        pos_pred = (self.W1.T @ self.h)[:2]
        
        if self.has_action:
            action = np.tanh(self.W_action @ self.h) * 0.5
            return pos_pred, action
        return pos_pred, None
    
    def predict_action(self, x):
        """For environment to call"""
        self.h = np.tanh(self.W1 @ x)
        if self.has_action:
            return np.tanh(self.W_action @ self.h) * 0.5
        return np.zeros(2)
    
    def update(self, x, y, action_target=None):
        pos_pred, action_pred = self.forward(x)
        
        # Position prediction loss
        pos_error = y - pos_pred
        mse = np.mean(pos_error ** 2)
        
        # Gradient
        if self.has_action and action_target is not None:
            # Combined loss: position + action prediction
            action_error = action_target - action_pred
            delta = np.mean(pos_error) * np.mean(self.h) - self.lam * np.sign(self.W1)
            delta_action = np.mean(action_error) * np.mean(self.h) - self.lam * np.sign(self.W_action)
            
            self.W1 += self.lr * delta
            self.W_action += self.lr * delta_action
        else:
            delta = np.mean(pos_error) * np.mean(self.h) - self.lam * np.sign(self.W1)
            self.W1 += self.lr * delta
        
        self.W1 = np.clip(self.W1, -10, 10)
        
        return mse
    
    def get_hidden(self, x):
        self.h = np.tanh(self.W1 @ x)
        return self.h.copy()


def train_passive(model, env, steps=3000):
    """Train passive model on environment-generated sequences"""
    losses = []
    for _ in range(steps):
        obs_seq, target_seq, vx, vy = env.generate_passive_sequence()
        x = obs_seq.flatten()
        y = target_seq[0]
        
        loss = model.update(x, y)
        losses.append(loss)
    
    return np.mean(losses[-500:])


def train_active(model, env, steps=3000):
    """Train active model on self-generated sequences"""
    losses = []
    for _ in range(steps):
        obs_seq, target_seq, vx, vy = env.generate_active_sequence(model)
        x = obs_seq.flatten()
        y = target_seq[0]
        
        # Active model also predicts what action led to this trajectory
        # Simplified: just predict next position
        loss = model.update(x, y)
        losses.append(loss)
    
    return np.mean(losses[-500:])


def evaluate_generalization(model, env_train, envs_test):
    """Test generalization across different alphas"""
    results = {}
    
    # Test on training environment
    obs, target, _ = env_train.generate_passive_sequence()
    x = obs.flatten()
    y = target[0]
    pos_pred, _ = model.forward(x)
    results['alpha_1.0'] = np.mean((pos_pred - y) ** 2)
    
    # Test on shifted environments
    for alpha, env in envs_test.items():
        obs, target, _ = env.generate_passive_sequence()
        x = obs.flatten()
        y = target[0]
        pos_pred, _ = model.forward(x)
        results[alpha] = np.mean((pos_pred - y) ** 2)
    
    # Gap calculation
    mse_e1 = results['alpha_1.0']
    mse_e2 = results['alpha_0.5']
    mse_e3 = results['alpha_2.0']
    
    gap = ((mse_e2 + mse_e3) / (2 * mse_e1)) if mse_e1 > 0 else float('inf')
    results['gap'] = gap
    
    return results


def linear_probe(model, env, n_samples=1000):
    """Linear probe: hidden state → velocity/position"""
    H, V, P = [], [], []
    
    for _ in range(n_samples):
        obs, target, vx, vy = env.generate_passive_sequence()
        x = obs.flatten()
        
        h = model.get_hidden(x)
        v = np.array([vx, vy])
        p = target[0]
        
        H.append(h)
        V.append(v)
        P.append(p)
    
    H, V, P = np.array(H), np.array(V), np.array(P)
    
    # Probe velocity
    m_v = LinearRegression()
    m_v.fit(H, V)
    v_pred = m_v.predict(H)
    r2_v = 1 - np.mean((V - v_pred) ** 2) / np.var(V)
    
    # Probe position
    m_p = LinearRegression()
    m_p.fit(H, P)
    p_pred = m_p.predict(H)
    r2_p = 1 - np.mean((P - p_pred) ** 2) / np.var(P)
    
    return {'r2_v': r2_v, 'r2_p': r2_p}


def ablation_test(model, env, n_samples=500):
    """Subspace ablation test"""
    # Get hidden states
    H, V, P = [], [], []
    for _ in range(n_samples):
        obs, target, vx, vy = env.generate_passive_sequence()
        x = obs.flatten()
        H.append(model.get_hidden(x))
        V.append([vx, vy])
        P.append(target[0])
    
    H, V, P = np.array(H), np.array(V), np.array(P)
    
    # Baseline MSE
    obs, target, _ = env.generate_passive_sequence()
    x = obs.flatten()
    y = target[0]
    pos_pred, _ = model.forward(x)
    baseline_mse = np.mean((pos_pred - y) ** 2)
    
    # Simple ablation: zero out random 50% of hidden units
    mask_full = np.ones(H.shape[1], dtype=bool)
    mask_half = np.random.rand(H.shape[1]) > 0.5
    
    # Test with half masked
    h_original = model.h.copy()
    
    # Measure importance via gradient
    model.h = np.zeros_like(model.h)
    obs, target, _ = env.generate_passive_sequence()
    x = obs.flatten()
    model.forward(x)
    
    # Get gradient magnitude for each hidden unit
    grad_mag = np.abs(model.W1 @ np.ones(6))
    important_units = grad_mag > np.percentile(grad_mag, 50)
    
    # Ablate important units
    model.W1_ablate = model.W1.copy()
    model.W1_ablate[important_units] = 0
    
    # Test MSE with ablation
    obs, target, _ = env.generate_passive_sequence()
    x = obs.flatten()
    h = np.tanh(model.W1_ablate @ x)
    pos_pred = (model.W1_ablate.T @ h)[:2]
    ablated_mse = np.mean((pos_pred - target[0]) ** 2)
    
    model.h = h_original
    
    return {
        'baseline_mse': baseline_mse,
        'ablated_mse': ablated_mse,
        'ratio': ablated_mse / baseline_mse if baseline_mse > 0 else float('inf')
    }


def control_test(model, env, target_pos=(10, 0), n_steps=20, has_action=False):
    """Control task: reach target"""
    x, y = 0, 0
    vx, vy = 0, 0
    
    trajectory = [(x, y)]
    
    for _ in range(n_steps):
        obs_seq = trajectory[-3:] if len(trajectory) >= 3 else [[x, y]] * 3
        obs_seq = np.array(obs_seq).flatten()
        
        if has_action:
            action = model.predict_action(obs_seq)
            x, y, vx, vy = env.step(x, y, vx, vy, action)
        else:
            # Passive: random action (no control)
            x += env.alpha * vx
            y += env.alpha * vy
        
        trajectory.append((x, y))
    
    # Distance to target
    final_pos = trajectory[-1]
    distance = np.sqrt((final_pos[0] - target_pos[0])**2 + (final_pos[1] - target_pos[1])**2)
    
    return {
        'final_pos': final_pos,
        'distance': distance,
        'trajectory': trajectory
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print("="*60)
print("ACTION VS PASSIVITY EXPERIMENT")
print("="*60)

# Parameters
n_hidden = 32
lam = 0.01
steps = 3000

# Environments
env_train = ActableMovingPointEnv(alpha=1.0)
envs_test = {
    'alpha_0.5': ActableMovingPointEnv(alpha=0.5),
    'alpha_2.0': ActableMovingPointEnv(alpha=2.0)
}

# Train PASSIVE group
print("\n[1] Training PASSIVE group...")
model_passive = PredictiveModel(n_hidden=n_hidden, has_action=False, lr=0.01, lam=lam)
loss_passive = train_passive(model_passive, env_train, steps=steps)
print(f"    Final MSE: {loss_passive:.4f}")

# Train ACTIVE group
print("\n[2] Training ACTIVE group...")
model_active = PredictiveModel(n_hidden=n_hidden, has_action=True, lr=0.01, lam=lam)
# Pre-train on passive data first
for _ in range(1000):
    obs, target, _, _ = env_train.generate_passive_sequence()
    model_active.update(obs.flatten(), target[0])
# Then train on active data
loss_active = train_active(model_active, env_train, steps=steps)
print(f"    Final MSE: {loss_active:.4f}")

# ============================================================
# EVALUATION
# ============================================================

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# 1. Generalization Gap
print("\n[1] Generalization Gap:")
gen_passive = evaluate_generalization(model_passive, env_train, envs_test)
gen_active = evaluate_generalization(model_active, env_train, envs_test)

print(f"    PASSIVE: alpha=1.0 MSE={gen_passive['alpha_1.0']:.4f}, gap={gen_passive['gap']:.4f}")
print(f"    ACTIVE:  alpha=1.0 MSE={gen_active['alpha_1.0']:.4f}, gap={gen_active['gap']:.4f}")

# 2. Linear Probe
print("\n[2] Linear Probe (R²):")
probe_passive = linear_probe(model_passive, env_train)
probe_active = linear_probe(model_active, env_train)

print(f"    PASSIVE: R²(velocity)={probe_passive['r2_v']:.3f}, R²(position)={probe_passive['r2_p']:.3f}")
print(f"    ACTIVE:  R²(velocity)={probe_active['r2_v']:.3f}, R²(position)={probe_active['r2_p']:.3f}")

# 3. Ablation
print("\n[3] Ablation Test:")
ablation_passive = ablation_test(model_passive, env_train)
ablation_active = ablation_test(model_active, env_train)

print(f"    PASSIVE: baseline={ablation_passive['baseline_mse']:.4f}, ablated={ablation_passive['ablated_mse']:.4f}, ratio={ablation_passive['ratio']:.2f}x")
print(f"    ACTIVE:  baseline={ablation_active['baseline_mse']:.4f}, ablated={ablation_active['ablated_mse']:.4f}, ratio={ablation_active['ratio']:.2f}x")

# 4. Control Test
print("\n[4] Control Task (reach target):")
control_passive = control_test(model_passive, env_train, has_action=False)
control_active = control_test(model_active, env_train, has_action=True)

print(f"    PASSIVE: distance to target = {control_passive['distance']:.2f}")
print(f"    ACTIVE:  distance to target = {control_active['distance']:.2f}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\nMetric                    | PASSIVE  | ACTIVE")
print("-"*50)
print(f"Generalization Gap        | {gen_passive['gap']:7.3f} | {gen_active['gap']:7.3f}")
print(f"R² (velocity)            | {probe_passive['r2_v']:7.3f} | {probe_active['r2_v']:7.3f}")
print(f"R² (position)            | {probe_passive['r2_p']:7.3f} | {probe_active['r2_p']:7.3f}")
print(f"Ablation ratio           | {ablation_passive['ratio']:7.2f}x | {ablation_active['ratio']:7.2f}x")
print(f"Control distance        | {control_passive['distance']:7.2f} | {control_active['distance']:7.2f}")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if probe_active['r2_v'] > probe_passive['r2_v'] + 0.1:
    print(">>> Active group shows HIGHER velocity encoding!")
else:
    print(">>> No significant velocity encoding difference")

if gen_active['gap'] < gen_passive['gap']:
    print(">>> Active group shows BETTER generalization!")
else:
    print(">>> No significant generalization difference")
