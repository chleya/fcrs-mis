"""
PHASE DIAGRAM VISUALIZATION
Representation vs Task Complexity
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Define regions
# Region 1: In-distribution Prediction (Trajectory wins)
ax.add_patch(plt.Rectangle((0, 0), 4, 3, fill=True, color='#90EE90', alpha=0.7))

# Region 2: Out-of-distribution Scaling (Object wins, after perception solved)
ax.add_patch(plt.Rectangle((4, 3), 4, 3, fill=True, color='#87CEEB', alpha=0.7))

# Region 3: Perception Bottleneck (both fail with pixels)
ax.add_patch(plt.Rectangle((4, 0), 4, 3, fill=True, color='#FFB6C1', alpha=0.7))

# Region 4: Reasoning/Causal (unknown)
ax.add_patch(plt.Rectangle((0, 3), 4, 3, fill=True, color='#DDA0DD', alpha=0.7))

# Labels
ax.text(2, 1.5, 'Trajectory\nDominates\n(In-distribution)', 
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(6, 1.5, 'Both Fail\n(Perception\nBottleneck)', 
        ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(6, 4.5, 'Object-Centric\nDominates\n(Scaling)', 
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(2, 4.5, '?\n(Reasoning/\nCausal)', 
        ha='center', va='center', fontsize=12, fontweight='bold')

# Axes
ax.axvline(x=4, color='black', linestyle='--', linewidth=2)
ax.axhline(y=3, color='black', linestyle='--', linewidth=2)

# Axis labels
ax.set_xlabel('Task Complexity / Object Count →', fontsize=14)
ax.set_ylabel('Representation Type →', fontsize=14)
ax.set_title('Representation Phase Diagram\nObject-Centric vs Trajectory Models', fontsize=16, fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(color='#90EE90', alpha=0.7, label='Trajectory Wins'),
    mpatches.Patch(color='#87CEEB', alpha=0.7, label='Object-Centric Wins'),
    mpatches.Patch(color='#FFB6C1', alpha=0.7, label='Perception Bottleneck'),
    mpatches.Patch(color='#DDA0DD', alpha=0.7, label='Unknown (Reasoning)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Set limits
ax.set_xlim(0, 8)
ax.set_ylim(0, 6)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('F:/fcrs_mis/phase_diagram.png', dpi=150, bbox_inches='tight')
print("Saved: F:/fcrs_mis/phase_diagram.png")

# Create detailed version
fig2, ax2 = plt.subplots(figsize=(14, 10))

# Background
ax2.set_facecolor('#f5f5f5')

# Draw regions with more detail
regions = [
    (0, 0, 5, 4, '#90EE90', 'Trajectory Zone\n(In-distribution Prediction)', 'Trajectory models\nexcel at prediction\nwithin training distribution'),
    (5, 4, 5, 4, '#87CEEB', 'Object-Centric Zone\n(Combinatorial Scaling)', 'Object models\nenable scaling to\nnew object counts'),
    (5, 0, 5, 4, '#FFB6C1', 'Perception Bottleneck\n(Pixel Input)', 'Object models fail\ndue to perception\n(not representation)'),
    (0, 4, 5, 4, '#DDA0DD', 'Reasoning Zone\n(Future Work)', 'Causal inference\nCounterfactuals\nPlanning'),
]

for x, y, w, h, color, title, desc in regions:
    ax2.add_patch(plt.Rectangle((x, y), w, h, fill=True, color=color, alpha=0.6, edgecolor='black', linewidth=2))
    ax2.text(x + w/2, y + h*0.7, title, ha='center', va='center', fontsize=13, fontweight='bold')
    ax2.text(x + w/2, y + h*0.35, desc, ha='center', va='center', fontsize=10)

# Axis lines
ax2.axvline(x=5, color='black', linestyle='--', linewidth=2)
ax2.axhline(y=4, color='black', linestyle='--', linewidth=2)

# Experiment markers
experiments = [
    (2, 2, 'L1: Variable Emergence', 'green'),
    (2, 2, 'Prediction Tasks', 'green'),
    (2, 5.5, 'Counterfactual', 'purple'),
    (7, 2, 'Pixel Scaling\n(2→6 objects)', 'red'),
    (7, 5.5, 'Coordinates\n(2→6 objects)', 'blue'),
]

for x, y, label, color in experiments:
    ax2.plot(x, y, 'o', markersize=15, color=color, markeredgecolor='black')
    ax2.text(x, y+0.5, label, ha='center', va='bottom', fontsize=9)

# Arrow showing key finding
ax2.annotate('', xy=(7, 5.5), xytext=(7, 2),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax2.text(7.5, 3.75, 'Perception\nBottleneck', ha='left', va='center', fontsize=10, rotation=90)

# Labels
ax2.set_xlabel('Task Complexity / Object Count', fontsize=14)
ax2.set_ylabel('Representation Abstraction Level', fontsize=14)
ax2.set_title('Representation Phase Diagram: When Do Object-Centric Models Win?', 
              fontsize=16, fontweight='bold', pad=20)

# Key finding box
textstr = 'Key Finding:\nObject-centric representations enable\ncombinatorial scaling, but their benefits\nare hidden by perception bottlenecks.'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.02, 0.02, textstr, transform=ax2.transAxes, fontsize=11,
        verticalalignment='bottom', bbox=props)

ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('F:/fcrs_mis/phase_diagram_detailed.png', dpi=150, bbox_inches='tight')
print("Saved: F:/fcrs_mis/phase_diagram_detailed.png")

print("\nDone! Two phase diagrams created.")
