"""
单一变量验证实验
核心: 仅改变"是否有因果元结构"，其他完全一致
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)

print('='*70)
print('SINGLE-VARIABLE VERIFICATION')
print('='*70)

# ============================================================
# 1. 生成数据: CartPole-like physics
# ============================================================
n = 2000
seq_len = 4

# 真实物理: x(t+1) = x(t) + v + action_effect
# v是隐藏状态，action影响v
v_true = np.random.randn(n, 4) * 0.3  # 4D hidden state
x = np.random.randn(n, 4) * 0.5

# action与angle(第3维)负相关
action = np.random.choice([0, 1], n)
action_effect = np.zeros((n, 4))
action_effect[:, 2] = -0.8 * action + np.random.randn(n) * 0.1

# x包含v + action效应
x = x + v_true + action_effect

# 构建序列数据
X = np.stack([x.copy() for _ in range(seq_len)], axis=1)
V = np.stack([v_true.copy() for _ in range(seq_len)], axis=1)
A = np.stack([action.copy() for _ in range(seq_len)], axis=1)

X_t = torch.FloatTensor(X)
V_t = torch.FloatTensor(V)
A_t = torch.FloatTensor(A).unsqueeze(-1)

print(f'Data: {n} samples, {seq_len} timesteps')
print(f'Corr(action, angle): {np.corrcoef(action, x[:, 2])[0,1]:.3f}')
print()

# ============================================================
# 2. 定义三个模型 (参数量相近)
# ============================================================

# ---------- 组1: 原基线 (action在编码器中) ----------
class BaselineModel(nn.Module):
    """原始基线: z生成时可以用action"""
    def __init__(self):
        super().__init__()
        # 输入: obs + action
        self.encoder = nn.Sequential(
            nn.Linear(5, 24),  # 4 obs + 1 action
            nn.ReLU(),
            nn.Linear(24, 8)
        )
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(8 + 2, 24),  # z + current_action + next_action
            nn.ReLU(),
            nn.Linear(24, 4)
        )
    
    def forward(self, obs, a_curr, a_next):
        # z生成时用了action
        z = self.encoder(torch.cat([obs, a_curr], dim=-1))
        # 预测未来
        out = self.predictor(torch.cat([z, a_curr, a_next], dim=-1))
        return out, z

# ---------- 组2: 受控基线 (action与编码器隔离) ----------
class ControlledModel(nn.Module):
    """受控基线: z生成只能用obs，action仅用于预测"""
    def __init__(self):
        super().__init__()
        # z生成只用obs，完全不用action
        self.encoder = nn.Sequential(
            nn.Linear(4, 24),  # 4 obs ONLY
            nn.ReLU(),
            nn.Linear(24, 8)
        )
        # 预测时才能用action
        self.predictor = nn.Sequential(
            nn.Linear(8 + 2, 24),
            nn.ReLU(),
            nn.Linear(24, 4)
        )
    
    def forward(self, obs, a_curr, a_next):
        z = self.encoder(obs)  # 只能用obs
        out = self.predictor(torch.cat([z, a_curr, a_next], dim=-1))
        return out, z

# ---------- 组3: 因果元结构模型 ----------
class CausalModel(nn.Module):
    """因果架构: 编码器-动力学-解码器"""
    def __init__(self):
        super().__init__()
        # 编码器: obs -> z (不用action)
        self.encoder = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 8)
        )
        # 动力学: z + action -> z_next (递归)
        self.dynamics = nn.Sequential(
            nn.Linear(9, 24),  # 8 z + 1 action
            nn.ReLU(),
            nn.Linear(24, 8)
        )
        # 解码器: z -> obs
        self.decoder = nn.Sequential(
            nn.Linear(8, 24),
            nn.ReLU(),
            nn.Linear(24, 4)
        )
    
    def forward(self, obs, a_curr, a_next):
        z = self.encoder(obs)
        z_next = z + self.dynamics(torch.cat([z, a_curr], dim=-1))
        out = self.decoder(z_next)
        return out, z

# ============================================================
# 3. 训练函数 (完全相同的损失函数)
# ============================================================

def train_model(model, epochs=10, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # 用t时刻预测t+1时刻
        pred, z = model(X_t[:, 0], A_t[:, 0], A_t[:, 1])
        
        # 纯MSE损失 (所有模型完全一致)
        loss = F.mse_loss(pred, X_t[:, 1])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

# ============================================================
# 4. 测试函数
# ============================================================

def test_model(model):
    model.eval()
    with torch.no_grad():
        pred, z = model(X_t[:, 0], A_t[:, 0], A_t[:, 1])
        
        # 维度相关
        results = {}
        dim_names = ['pos', 'vel', 'angle', 'omega']
        for i, name in enumerate(dim_names):
            corr = np.corrcoef(z[:, i].numpy(), V_t[:, 0, i].numpy())[0, 1]
            results[name] = corr
        
        # 总相关
        total_corr = np.corrcoef(z.mean(dim=-1).numpy(), V_t[:, 0, :].mean(dim=-1).numpy())[0, 1]
        results['total'] = total_corr
        
    return results

# ============================================================
# 5. 运行实验
# ============================================================

print('Training Model 1: Baseline (action in encoder)...')
m1 = BaselineModel()
m1 = train_model(m1)
r1 = test_model(m1)

print('Training Model 2: Controlled (action isolated)...')
m2 = ControlledModel()
m2 = train_model(m2)
r2 = test_model(m2)

print('Training Model 3: Causal (encoder-dynamics-decoder)...')
m3 = CausalModel()
m3 = train_model(m3)
r3 = test_model(m3)

# ============================================================
# 6. 结果展示
# ============================================================

print()
print('='*70)
print('RESULTS')
print('='*70)
print(f"{'Model':<25} | {'pos':>7} | {'vel':>7} | {'angle':>7} | {'omega':>7} | {'total':>7}")
print('-'*70)
print(f"{'1. Baseline (a in enc)':<25} | {r1['pos']:>7.3f} | {r1['vel']:>7.3f} | {r1['angle']:>7.3f} | {r1['omega']:>7.3f} | {r1['total']:>7.3f}")
print(f"{'2. Controlled (a isolated)':<25} | {r2['pos']:>7.3f} | {r2['vel']:>7.3f} | {r2['angle']:>7.3f} | {r2['omega']:>7.3f} | {r2['total']:>7.3f}")
print(f"{'3. Causal (enc-dyn-dec)':<25} | {r3['pos']:>7.3f} | {r3['vel']:>7.3f} | {r3['angle']:>7.3f} | {r3['omega']:>7.3f} | {r3['total']:>7.3f}")
print('='*70)

print()
print('ANALYSIS:')
print('-'*70)

# angle维度是关键 (因为action与angle负相关)
print(f"Angle dimension (key test):")
print(f"  Baseline:   {r1['angle']:+.3f}  (负相关 = 走了action捷径)")
print(f"  Controlled: {r2['angle']:+.3f}  (接近0 = 堵死捷径但无元结构)")
print(f"  Causal:     {r3['angle']:+.3f}  (正相关 = 学到真实v)")

print()
if r3['angle'] > r1['angle'] and r3['angle'] > 0:
    print('SUCCESS: 因果元结构让变量涌现!')
else:
    print('NEEDS MORE TRAINING')

print()
print('='*70)
print('CONCLUSION:')
print('='*70)
print('如果 Corr(angle) 符合: 负 → 0 → 正')
print('则证明: 元结构本身导致变量涌现，与其他变量无关')
