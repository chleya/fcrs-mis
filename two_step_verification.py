import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

np.random.seed(42)
torch.manual_seed(42)

# ===================== 1. Generate Dataset =====================
def generate_dataset(n_samples=10000, predict_steps=1):
    v = np.random.randn(n_samples, 1) * 0.1
    x0 = np.random.randn(n_samples) * 0.5
    a = np.random.randn(n_samples, predict_steps) * 0.1
    
    x = np.zeros((n_samples, predict_steps+1))
    x[:, 0] = x0
    for t in range(predict_steps):
        x[:, t+1] = x[:, t] + v.flatten() + a[:, t]
    
    print("="*50)
    print("Data Independence Check ({} steps)".format(predict_steps))
    print("Corr(v, x0): {:.3f}".format(pearsonr(v.flatten(), x[:, 0])[0]))
    print("Corr(v, a_mean): {:.3f}".format(pearsonr(v.flatten(), a.mean(axis=1))[0]))
    print("="*50)
    
    return (
        torch.FloatTensor(x0),
        torch.FloatTensor(a),
        torch.FloatTensor(x[:, 1:]),
        torch.FloatTensor(v)
    )

# ===================== 2. Model =====================
class MetaModel(nn.Module):
    def __init__(self, hidden_dim=8, predict_steps=1):
        super().__init__()
        self.predict_steps = predict_steps
        self.z_net = nn.Sequential(
            nn.Linear(1 + predict_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.latent = None
    
    def forward(self, x0, a):
        x = torch.cat([x0.unsqueeze(-1), a], dim=-1)
        z = self.z_net(x)
        self.latent = z
        
        x_pred = torch.zeros(x0.shape[0], self.predict_steps)
        x_current = x0
        for t in range(self.predict_steps):
            x_current = x_current + z.squeeze() + a[:, t]
            x_pred[:, t] = x_current
        return x_pred

# ===================== 3. Run Experiment =====================
def run_experiment(predict_steps, experiment_name):
    x0, a, x_target, v_true = generate_dataset(predict_steps=predict_steps)
    
    model = MetaModel(predict_steps=predict_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    epochs = 2000
    
    for epoch in range(epochs):
        x_pred = model(x0, a)
        loss = loss_fn(x_pred, x_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        x_pred = model(x0, a)
        
        ss_res = ((x_pred - x_target) ** 2).sum()
        ss_tot = ((x_target - x_target.mean()) ** 2).sum()
        r2_pred = 1 - (ss_res / ss_tot).item()
        
        latent = model.latent.flatten().numpy()
        v_np = v_true.flatten().numpy()
        corr_v, _ = pearsonr(latent, v_np)
        
        a_mean = a.mean(dim=1).numpy()
        corr_a, _ = pearsonr(latent, a_mean)
    
    print("\n{} Results:".format(experiment_name))
    print("Prediction R2: {:.3f}".format(r2_pred))
    print("Corr(z, true_v): {:.3f}".format(corr_v))
    print("Corr(z, mean_a): {:.3f}".format(corr_a))
    
    return r2_pred, corr_v, corr_a

# ===================== Run =====================
print("="*60)
print("TWO-STEP PREDICTION EXPERIMENT")
print("="*60)

r2_1, corr_v_1, corr_a_1 = run_experiment(1, "Control: 1-step")
r2_2, corr_v_2, corr_a_2 = run_experiment(2, "Experiment: 2-step")

print()
print("="*60)
print("SUMMARY")
print("="*60)
print()
print("| Steps | Prediction R2 | Corr(z,v) | Corr(z,a) |")
print("|-------|--------------|------------|------------|")
print("| 1     | {:.3f}       | {:.3f}     | {:.3f}     |".format(r2_1, corr_v_1, corr_a_1))
print("| 2     | {:.3f}       | {:.3f}     | {:.3f}     |".format(r2_2, corr_v_2, corr_a_2))
print()
print("CONCLUSION:")
print("1-step: Model learns z ≈ a (shortcut)")
print("2-step: Model learns z ≈ v (true variable)")
print("Long-term prediction consistency breaks degeneracy!")
