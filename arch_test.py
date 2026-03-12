import numpy as np
import torch
import torch.nn as nn

# ========================
# 极简CPU实验：元结构验证
# ========================

np.random.seed(42)
torch.manual_seed(42)

# 生成数据（1D质点运动 + action）
def generate_data(n=1000, seq_len=10):
    # Position: cumulative sum of random walk
    x = np.cumsum(np.random.randn(n, seq_len) * 0.1, axis=1)
    
    # Action: random force applied after frame 5
    a = np.random.randn(n, seq_len) * 0.2
    x[:, 5:] += a[:, 5:]
    
    # True velocity: difference between consecutive positions
    v = np.diff(x, axis=1)
    
    return (torch.FloatTensor(x), 
            torch.FloatTensor(v), 
            torch.FloatTensor(a))

# 模型A: 白板MLP（无预设结构）
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x, a=None):
        return self.f(x)

# 模型B: 仅空间结构（object-centric）
class SpaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj = nn.Linear(1, 4)  # 空间嵌入
        self.pred = nn.Linear(4, 1)
    def forward(self, x, a=None):
        return self.pred(self.obj(x))

# 模型C: 仅时间结构（递归状态）
class TimeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = nn.Linear(1, 4)  # 状态递推
        self.pred = nn.Linear(4, 1)
    def forward(self, x, a=None):
        return self.pred(self.state(x))

# 模型D: 仅因果结构（action -> effect）
class CausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Linear(1, 4)   # state encoding
        self.a = nn.Linear(1, 4)   # action encoding (因果关键)
        self.pred = nn.Linear(4, 1)
    def forward(self, x, a):
        # 因果: state += action effect
        s = self.s(x) + self.a(a)
        return self.pred(s)

# 模型E: 完整结构（空间+时间+因果）
class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj = nn.Linear(1, 4)   # 空间
        self.state = nn.Linear(4, 4)  # 时间
        self.action = nn.Linear(1, 4) # 因果
        self.pred = nn.Linear(4, 1)
    def forward(self, x, a):
        s = self.obj(x)
        s = self.state(s) + self.action(a)
        return self.pred(s)

# 训练并计算R²
def train_and_evaluate(model_class, has_action=False):
    x, v, a = generate_data()
    
    model = model_class()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练
    for _ in range(2000):
        if has_action:
            pred = model(x[:, 0:1], a[:, 0:1])
        else:
            pred = model(x[:, 0:1])
        
        target = x[:, 1:2]  # 预测下一帧位置
        loss = nn.MSELoss()(pred, target)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # 提取latent并计算R²(velocity)
    with torch.no_grad():
        if has_action:
            # 获取state latent
            if hasattr(model, 's'):
                latent = model.s(x[:, 0:1]) + model.a(a[:, 0:1])
            elif hasattr(model, 'obj'):
                latent = model.obj(x[:, 0:1])
            else:
                latent = model.state(x[:, 0:1])
        else:
            if hasattr(model, 'obj'):
                latent = model.obj(x[:, 0:1])
            elif hasattr(model, 'state'):
                latent = model.state(x[:, 0:1])
            else:
                latent = x[:, 0:1] * 0 + 1  # fallback
        
        # R² vs velocity
        true_v = v[:, 0:1]
        ss_res = ((latent - true_v) ** 2).sum()
        ss_tot = ((true_v - true_v.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
    
    return float(r2)

# ========================
# 运行实验
# ========================
print("="*60)
print("实验: 哪种预设结构导致变量(velocity)涌现?")
print("="*60)
print()

# 所有模型都用action（干预）
models = [
    ("A: 白板MLP", MLP, True),
    ("B: 仅空间", SpaceModel, True),
    ("C: 仅时间", TimeModel, True),
    ("D: 仅因果", CausalModel, True),
    ("E: 完整结构", FullModel, True),
]

results = {}
for name, model_class, has_action in models:
    r2 = train_and_evaluate(model_class, has_action)
    results[name] = r2
    print(f"{name}: R² = {r2:.3f}")

print()
print("="*60)
print("结论:")
print("="*60)
print("R² > 0.6 = 变量强涌现")
print("R² < 0.2 = 无变量涌现")
print()
print(f"白板MLP R²={results['A: 白板MLP']:.3f} → {'❌ 无涌现' if results['A: 白板MLP'] < 0.2 else '✅ 有涌现'}")
print(f"仅因果  R²={results['D: 仅因果']:.3f} → {'❌ 无涌现' if results['D: 仅因果'] < 0.6 else '✅ 强涌现'}")
print(f"完整结构 R²={results['E: 完整结构']:.3f} → {'❌ 无涌现' if results['E: 完整结构'] < 0.6 else '✅ 强涌现'}")
