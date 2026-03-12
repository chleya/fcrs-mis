import numpy as np

np.random.seed(42)

print('='*60)
print('EXPERIMENT 2: Latent Sensitivity to Action')
print('='*60)

class Env:
    def reset(self):
        self.x = np.random.uniform(-5, 5)
        self.v = np.random.uniform(-1, 1)
        self.friction = 0.9
    def step(self, action):
        self.v += action
        self.v *= self.friction
        self.x += self.v
        return self.x

latent_dim = 4
W = np.random.randn(latent_dim, 2) * 0.1

for _ in range(3000):
    e = Env()
    e.reset()
    action = np.random.uniform(-1, 1)
    obs = np.array([e.x, action])
    x_next = e.step(action)
    h = np.tanh(W @ obs)
    pred = np.sum(W.T * h)
    loss = x_next - pred
    W += 0.01 * (loss * np.mean(h) - 0.01 * np.sign(W))

# Fix x, vary action
x_fixed = 2.0
actions = [-1.0, 0.0, 1.0]

print('Fix x=2.0, vary action:')
for a in actions:
    obs = np.array([x_fixed, a])
    h = np.tanh(W @ obs)
    print(f'  action={a:.1f}: latent={[f"{v:.2f}" for v in h]}')

h_min = np.tanh(W @ np.array([x_fixed, -1.0]))
h_max = np.tanh(W @ np.array([x_fixed, 1.0]))
delta = h_max - h_min

print(f'  Delta (action -1 to 1): {[f"{v:.2f}" for v in delta]}')
print('')
print('INTERPRETATION:')
print('If velocity is explicit: one dimension tracks action linearly')
print('If implicit: latent changes distributed')
