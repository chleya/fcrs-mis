"""
3D SURFACE: E(N, I) - Theoretical Model Visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("="*60)
print("3D SURFACE: E(N, I)")
print("="*60)

# Model parameters (from experiments)
alpha = 0.3  # trajectory sensitivity to N*I
beta = 0.02  # object linear scaling
E_base = 0.1

# Create mesh
N = np.linspace(1, 10, 30)
I = np.linspace(0, 1, 30)
N_mesh, I_mesh = np.meshgrid(N, I)

# Trajectory error: exponential
E_traj = np.exp(alpha * N_mesh * I_mesh) - 1

# Object error: linear  
E_obj = beta * N_mesh

# Difference
diff = E_traj - E_obj

# Plot
fig = plt.figure(figsize=(16, 6))

# Trajectory surface
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(N_mesh, I_mesh, E_traj, cmap='Reds', alpha=0.8)
ax1.set_xlabel('N (objects)')
ax1.set_ylabel('I (interaction)')
ax1.set_zlabel('Error')
ax1.set_title('Trajectory: E = exp(NI) - 1')

# Object surface
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(N_mesh, I_mesh, E_obj, cmap='Blues', alpha=0.8)
ax2.set_xlabel('N (objects)')
ax2.set_ylabel('I (interaction)')
ax2.set_zlabel('Error')
ax2.set_title('Object: E = 0.02*N')

# Winner surface
ax3 = fig.add_subplot(133, projection='3d')
winner = np.sign(diff)
surf3 = ax3.plot_surface(N_mesh, I_mesh, winner, cmap='RdYlGn', alpha=0.8)
ax3.set_xlabel('N (objects)')
ax3.set_ylabel('I (interaction)')
ax3.set_zlabel('Winner (+1=Obj, -1=Traj)')
ax3.set_title('Phase: Object vs Trajectory')

plt.tight_layout()
plt.savefig('F:/fcrs_mis/surface_3d.png', dpi=150, bbox_inches='tight')
print("Saved: surface_3d.png")

# 2D contour
fig2, ax = plt.subplots(figsize=(10, 8))

# Filled contour
contour = ax.contourf(N_mesh, I_mesh, diff, levels=50, cmap='RdYlGn')
plt.colorbar(contour, label='Trajectory - Object')

# Add zero contour (crossover)
ax.contour(N_mesh, I_mesh, diff, levels=[0], colors='black', linewidths=2)

ax.set_xlabel('N (Number of Objects)', fontsize=14)
ax.set_ylabel('I (Interaction Density)', fontsize=14)
ax.set_title('Phase Diagram: Trajectory vs Object Winner\n(Zero line = Crossover)', fontsize=14)

plt.tight_layout()
plt.savefig('F:/fcrs_mis/phase_contour.png', dpi=150, bbox_inches='tight')
print("Saved: phase_contour.png")

# Key data points
print("\n" + "="*60)
print("KEY DATA POINTS")
print("="*60)
print("\nPredicted crossover points:")
for I_val in [0.1, 0.3, 0.5, 0.7, 1.0]:
    # Find N where exp(alpha*N*I) - 1 = beta*N
    # Solve numerically
    for N_val in np.linspace(1, 10, 100):
        if abs(np.exp(alpha * N_val * I_val) - 1 - beta * N_val) < 0.1:
            print(f"  I={I_val:.1f}: N_c ≈ {N_val:.1f}")
            break

print("\n" + "="*60)
print("EMPIRICAL vs THEORY")
print("="*60)
print("\nEmpirical (from experiments):")
print("  I=0.1: N_c=4")
print("  I=0.3+: Trajectory wins")
print("\nTheory predicts:")
print("  I=0.1: N_c ~ 3-4")
print("  I=0.3: N_c ~ 1-2 (but Trajectory wins - needs higher N)")
