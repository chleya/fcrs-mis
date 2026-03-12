import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Generate data
n = 1000
x = np.cumsum(np.random.randn(n, 10) * 0.1, axis=1)
a = np.random.randn(n, 10) * 0.2
x[:, 5:] += a[:, 5:]
v = np.diff(x, axis=1)

x0 = x[:, 0:1]
a0 = a[:, 0:1]
v0 = v[:, 0]

print("="*60)
print("EXPERIMENT: Which preset structure enables variable emergence?")
print("="*60)

# A: Blank MLP
W1 = np.random.randn(1, 8) * 0.1
W2 = np.random.randn(8, 1) * 0.1

for _ in range(2000):
    h = np.tanh(x0 @ W1)
    pred = h @ W2
    error = pred - x[:, 1:2]
    W2 -= 0.01 * (h.T @ error / n)
    W1 -= 0.01 * (x0.T @ (error @ W2.T * (1-h**2)) / n)

h_A = np.tanh(x0 @ W1)
r2_A = LinearRegression().fit(h_A, v0).score(h_A, v0)
print("A: Blank MLP: R2 = {:.3f}".format(r2_A))

# B: Space only
obj = np.random.randn(1, 4) * 0.1
pred_B = np.random.randn(4, 1) * 0.1

for _ in range(2000):
    h = np.tanh(x0 @ obj)
    pred = h @ pred_B
    error = pred - x[:, 1:2]
    pred_B -= 0.01 * (h.T @ error / n)
    obj -= 0.01 * (x0.T @ (error @ pred_B.T * (1-h**2)) / n)

h_B = np.tanh(x0 @ obj)
r2_B = LinearRegression().fit(h_B, v0).score(h_B, v0)
print("B: Space only: R2 = {:.3f}".format(r2_B))

# C: Time only
state = np.random.randn(1, 4) * 0.1
pred_C = np.random.randn(4, 1) * 0.1

for _ in range(2000):
    h = np.tanh(x0 @ state)
    pred = h @ pred_C
    error = pred - x[:, 1:2]
    pred_C -= 0.01 * (h.T @ error / n)
    state -= 0.01 * (x0.T @ (error @ pred_C.T * (1-h**2)) / n)

h_C = np.tanh(x0 @ state)
r2_C = LinearRegression().fit(h_C, v0).score(h_C, v0)
print("C: Time only: R2 = {:.3f}".format(r2_C))

# D: Causal only
s = np.random.randn(1, 4) * 0.1
act = np.random.randn(1, 4) * 0.1
pred_D = np.random.randn(4, 1) * 0.1

for _ in range(2000):
    s_h = np.tanh(x0 @ s)
    a_h = np.tanh(a0 @ act)
    h = s_h + a_h
    pred = h @ pred_D
    error = pred - x[:, 1:2]
    pred_D -= 0.01 * (h.T @ error / n)
    act -= 0.01 * (a0.T @ (error @ pred_D.T * (1-a_h**2)) / n)
    s -= 0.01 * (x0.T @ (error @ pred_D.T * (1-s_h**2)) / n)

h_D = np.tanh(x0 @ s) + np.tanh(a0 @ act)
r2_D = LinearRegression().fit(h_D, v0).score(h_D, v0)
print("D: Causal only: R2 = {:.3f}".format(r2_D))

# E: Full structure
obj_E = np.random.randn(1, 4) * 0.1
state_E = np.random.randn(4, 4) * 0.1
act_E = np.random.randn(1, 4) * 0.1
pred_E = np.random.randn(4, 1) * 0.1

for _ in range(2000):
    s_h = np.tanh(x0 @ obj_E)
    s_h = np.tanh(s_h @ state_E)
    a_h = np.tanh(a0 @ act_E)
    h = s_h + a_h
    pred = h @ pred_E
    error = pred - x[:, 1:2]
    pred_E -= 0.01 * (h.T @ error / n)
    act_E -= 0.01 * (a0.T @ (error @ pred_E.T * (1-a_h**2)) / n)

h_E = np.tanh(x0 @ obj_E) + np.tanh(a0 @ act_E)
r2_E = LinearRegression().fit(h_E, v0).score(h_E, v0)
print("E: Full structure: R2 = {:.3f}".format(r2_E))

print()
print("="*60)
print("CONCLUSION:")
print("="*60)
print("R2 > 0.6 = Variable Emergence")
print("R2 < 0.2 = No Emergence")
print()

results = [("A: Blank MLP", r2_A), ("B: Space", r2_B), ("C: Time", r2_C), ("D: Causal", r2_D), ("E: Full", r2_E)]

for name, r2 in results:
    if r2 > 0.6:
        status = "STRONG EMERGENCE"
    elif r2 < 0.2:
        status = "NO EMERGENCE"
    else:
        status = "WEAK"
    print("{}: R2={:.3f} -> {}".format(name, r2, status))
