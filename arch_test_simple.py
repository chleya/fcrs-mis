import numpy as np
from sklearn.linear_model import LinearRegression

# ========================
# 极简实验：元结构验证
# ========================

np.random.seed(42)

# 生成数据
n = 1000
x = np.cumsum(np.random.randn(n, 10) * 0.1, axis=1)
a = np.random.randn(n, 10) * 0.2
x[:, 5:] += a[:, 5:]
v = np.diff(x, axis=1)

x0 = x[:, 0:1]  # 第一帧
a0 = a[:, 0:1]  # 第一个action
v0 = v[:, 0]    # 第一个速度

print("="*60)
print("实验: 哪种预设结构导致变量(velocity)涌现?")
print("="*60)

# ========================
# 模型A: 白板MLP
# ========================
W1 = np.random.randn(8, 1) * 0.1
W2 = np.random.randn(1, 8) * 0.1

for _ in range(2000):
    h = np.tanh(x0 @ W1)
    pred = h @ W2
    error = pred - x[:, 1:2]
    W2 -= 0.01 * (error.T @ h / n)
    W1 -= 0.01 * ((error @ W2) * (1-h**2)).T @ x0 / n

h_A = np.tanh(x0 @ W1)
r2_A = LinearRegression().fit(h_A, v0).score(h_A, v0)
print(f"A: 白板MLP: R² = {r2_A:.3f}")

# ========================
# 模型B: 仅空间
# ========================
obj = np.random.randn(4, 1) * 0.1
pred_B = np.random.randn(1, 4) * 0.1

for _ in range(2000):
    h = np.tanh(x0 @ obj)
    pred = h @ pred_B
    error = pred - x[:, 1:2]
    pred_B -= 0.01 * (error.T @ h / n)
    obj -= 0.01 * ((error @ pred_B) * (1-h**2)).T @ x0 / n

h_B = np.tanh(x0 @ obj)
r2_B = LinearRegression().fit(h_B, v0).score(h_B, v0)
print(f"B: 仅空间: R² = {r2_B:.3f}")

# ========================
# 模型C: 仅时间
# ========================
state = np.random.randn(4, 1) * 0.1
pred_C = np.random.randn(1, 4) * 0.1

for _ in range(2000):
    h = np.tanh(x0 @ state)
    pred = h @ pred_C
    error = pred - x[:, 1:2]
    pred_C -= 0.01 * (error.T @ h / n)
    state -= 0.01 * ((error @ pred_C) * (1-h**2)).T @ x0 / n

h_C = np.tanh(x0 @ state)
r2_C = LinearRegression().fit(h_C, v0).score(h_C, v0)
print(f"C: 仅时间: R² = {r2_C:.3f}")

# ========================
# 模型D: 仅因果
# ========================
s = np.random.randn(4, 1) * 0.1
act = np.random.randn(4, 1) * 0.1
pred_D = np.random.randn(1, 4) * 0.1

for _ in range(2000):
    s_h = np.tanh(x0 @ s)
    a_h = np.tanh(a0 @ act)
    h = s_h + a_h
    pred = h @ pred_D
    error = pred - x[:, 1:2]
    pred_D -= 0.01 * (error.T @ h / n)
    act -= 0.01 * ((error @ pred_D) * (1-a_h**2)).T @ a0 / n
    s -= 0.01 * ((error @ pred_D) * (1-s_h**2)).T @ x0 / n

h_D = np.tanh(x0 @ s) + np.tanh(a0 @ act)
r2_D = LinearRegression().fit(h_D, v0).score(h_D, v0)
print(f"D: 仅因果: R² = {r2_D:.3f}")

# ========================
# 模型E: 完整结构
# ========================
obj_E = np.random.randn(4, 1) * 0.1
state_E = np.random.randn(4, 4) * 0.1
act_E = np.random.randn(4, 1) * 0.1
pred_E = np.random.randn(1, 4) * 0.1

for _ in range(2000):
    s_h = np.tanh(x0 @ obj_E)
    s_h = np.tanh(s_h @ state_E)
    a_h = np.tanh(a0 @ act_E)
    h = s_h + a_h
    pred = h @ pred_E
    error = pred - x[:, 1:2]
    pred_E -= 0.01 * (error.T @ h / n)
    act_E -= 0.01 * ((error @ pred_E) * (1-a_h**2)).T @ a0 / n

h_E = np.tanh(x0 @ obj_E) + np.tanh(a0 @ act_E)
r2_E = LinearRegression().fit(h_E, v0).score(h_E, v0)
print(f"E: 完整结构: R² = {r2_E:.3f}")

print()
print("="*60)
print("结论:")
print("="*60)
print("R² > 0.6 = 变量强涌现")
print("R² < 0.2 = 无变量涌现")
print()

results = [("A: 白板MLP", r2_A), ("B: 仅空间", r2_B), ("C: 仅时间", r2_C), ("D: 仅因果", r2_D), ("E: 完整结构", r2_E)]

for name, r2 in results:
    status = "✅ 强涌现" if r2 > 0.6 else "❌ 无涌现" if r2 < 0.2 else "⚠️ 弱涌现"
    print(f"{name}: R²={r2:.3f} → {status}")
