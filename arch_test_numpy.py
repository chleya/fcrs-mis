import numpy as np

# ========================
# 极简CPU实验：元结构验证（纯numpy）
# ========================

np.random.seed(42)

# 生成数据（1D质点运动 + action）
def generate_data(n=1000, seq_len=10):
    # Position: cumulative sum of random walk
    x = np.cumsum(np.random.randn(n, seq_len) * 0.1, axis=1)
    
    # Action: random force applied after frame 5
    a = np.random.randn(n, seq_len) * 0.2
    x[:, 5:] += a[:, 5:]
    
    # True velocity: difference between consecutive positions
    v = np.diff(x, axis=1)
    
    return x, v, a

# 模型类
class Model:
    def __init__(self, arch):
        self.arch = arch
        np.random.seed(42)
        
        if arch == 'A_mlp':
            self.W1 = np.random.randn(8, 1) * 0.1
            self.W2 = np.random.randn(1, 8) * 0.1
            
        elif arch == 'B_space':
            self.obj = np.random.randn(4, 1) * 0.1
            self.pred = np.random.randn(1, 4) * 0.1
            
        elif arch == 'C_time':
            self.state = np.random.randn(4, 1) * 0.1
            self.pred = np.random.randn(1, 4) * 0.1
            
        elif arch == 'D_causal':
            self.s = np.random.randn(4, 1) * 0.1
            self.a = np.random.randn(4, 1) * 0.1
            self.pred = np.random.randn(1, 4) * 0.1
            
        elif arch == 'E_full':
            self.obj = np.random.randn(4, 1) * 0.1
            self.state = np.random.randn(4, 4) * 0.1
            self.action = np.random.randn(4, 1) * 0.1
            self.pred = np.random.randn(1, 4) * 0.1
    
    def forward(self, x, a=None):
        # x shape: (n, 1)
        
        if self.arch == 'A_mlp':
            h = np.tanh(self.W1 @ x.T).T
            return (self.W2 @ h.T).T
        
        elif self.arch == 'B_space':
            h = np.tanh(self.obj @ x.T).T
            return (self.pred @ h.T).T
        
        elif self.arch == 'C_time':
            h = np.tanh(self.state @ x.T).T
            return (self.pred @ h.T).T
        
        elif self.arch == 'D_causal':
            s = np.tanh(self.s @ x.T).T
            if a is not None:
                a_h = np.tanh(self.a @ a.T).T
                s = s + a_h
            return (self.pred @ s.T).T
        
        elif self.arch == 'E_full':
            s = np.tanh(self.obj @ x.T).T
            s = np.tanh((s @ self.state))
            if a is not None:
                a_h = np.tanh(self.a @ a.T).T
                s = s + a_h
            return (self.pred @ s.T).T
    
    def get_latent(self, x, a=None):
        if self.arch == 'A_mlp':
            return np.tanh(self.W1 @ x.T).T
        
        elif self.arch == 'B_space':
            return np.tanh(self.obj @ x.T).T
        
        elif self.arch == 'C_time':
            return np.tanh(self.state @ x.T).T
        
        elif self.arch == 'D_causal':
            s = np.tanh(self.s @ x.T).T
            if a is not None:
                a_h = np.tanh(self.a @ a.T).T
                s = s + a_h
            return s
        
        elif self.arch == 'E_full':
            s = np.tanh(self.obj @ x.T).T
            s = np.tanh((s @ self.state))
            if a is not None:
                a_h = np.tanh(self.a @ a.T).T
                s = s + a_h
            return s
    
    def train(self, x, v, a, epochs=2000, lr=0.01):
        # x: (n, seq_len), v: (n, seq_len-1), a: (n, seq_len)
        x0 = x[:, 0:1]  # first frame
        x1 = x[:, 1:2]  # next frame
        a0 = a[:, 0:1]   # first action
        
        for ep in range(epochs):
            if self.arch in ['A_mlp', 'B_space', 'C_time']:
                pred = self.forward(x0)
            else:
                pred = self.forward(x0, a0)
            
            error = pred - x1
            
            # 梯度更新
            if self.arch == 'A_mlp':
                h = np.tanh(self.W1 @ x0.T).T
                grad_W2 = (error.T @ h) / len(x)
                grad_W1 = ((error @ self.W2.T) * (1-h**2)).T @ x0 / len(x)
                self.W2 -= lr * grad_W2
                self.W1 -= lr * grad_W1
                
            elif self.arch == 'D_causal':
                s = np.tanh(self.s @ x0.T).T
                a_h = np.tanh(self.a @ a0.T).T
                s_final = s + a_h
                
                grad_pred = (error.T @ s_final) / len(x)
                grad_a = ((error @ self.pred.T) * (1 - a_h**2)).T @ a0 / len(x)
                grad_s = ((error @ self.pred.T) * (1 - s**2)).T @ x0 / len(x)
                
                self.pred -= lr * grad_pred
                self.a -= lr * grad_a
                self.s -= lr * grad_s
            
            elif self.arch == 'E_full':
                s = np.tanh(self.obj @ x0.T).T
                s = np.tanh((s @ self.state))
                a_h = np.tanh(self.a @ a0.T).T
                s_final = s + a_h
                
                grad_pred = (error.T @ s_final) / len(x)
                grad_a = ((error @ self.pred.T) * (1 - a_h**2)).T @ a0 / len(x)
                
                self.pred -= lr * grad_pred
                self.a -= lr * grad_a
            
            # 简化其他模型训练
            elif self.arch == 'B_space':
                h = np.tanh(self.obj @ x0.T).T
                grad_pred = (error.T @ h) / len(x)
                grad_obj = ((error @ self.pred.T) * (1-h**2)).T @ x0 / len(x)
                self.pred -= lr * grad_pred
                self.obj -= lr * grad_obj
                
            elif self.arch == 'C_time':
                h = np.tanh(self.state @ x0.T).T
                grad_pred = (error.T @ h) / len(x)
                grad_state = ((error @ self.pred.T) * (1-h**2)).T @ x0 / len(x)
                self.pred -= lr * grad_pred
                self.state -= lr * grad_state

# ========================
# 运行实验
# ========================

print("="*60)
print("实验: 哪种预设结构导致变量(velocity)涌现?")
print("="*60)

x, v, a = generate_data()
x0 = x[:, 0:1]
a0 = a[:, 0:1]

results = {}

architectures = [
    ("A: 白板MLP", "A_mlp"),
    ("B: 仅空间", "B_space"),
    ("C: 仅时间", "C_time"),
    ("D: 仅因果", "D_causal"),
    ("E: 完整结构", "E_full"),
]

for name, arch in architectures:
    model = Model(arch)
    model.train(x, v, a)
    
    # 计算R²(velocity)
    with np.no_grad():
        latent = model.get_latent(x0, a0)
        true_v = v[:, 0:1]
        
        ss_res = ((latent - true_v) ** 2).sum()
        ss_tot = ((true_v - true_v.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
    
    results[name] = r2
    print(f"{name}: R² = {r2:.3f}")

print()
print("="*60)
print("结论:")
print("="*60)
print("R² > 0.6 = 变量强涌现")
print("R² < 0.2 = 无变量涌现")
print()

for name, r2 in results.items():
    status = "✅ 强涌现" if r2 > 0.6 else "❌ 无涌现" if r2 < 0.2 else "⚠️ 弱涌现"
    print(f"{name}: R²={r2:.3f} → {status}")
