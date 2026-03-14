# Structure-Capability Law - Research Summary

## L1-L7 Complete Table

| Level | Capability | Required Structure | Formula / Critical Point |
|-------|------------|-------------------|------------------------|
| L1 | Causal Variable Emergence | Encoder Isolation + λ>0.01 | λ_c = 0.01 |
| L2 | Object Identity Tracking | Temporal + Binding | λ > 0.05 |
| L3 | Compositional Generalization | Factorized MLP | N_c ≈ 4 |
| L4 | Causal Robustness | Random Noise + Causal Training | +24% improvement |
| L5 | Goal-Directed Planning | Planning Head + Binding | Success > baseline |
| L6 | Scaling Critical Point | Interaction Density + Binding | N_c ≈ 4 |
| **L7** | **Inter-Object Relational Reasoning** | **Graph Density ≥0.3 + Binding λ≥0.1** | **λ × Density > 0.03** |

## New Findings from 3 Directions

### 1. Binding Strength Sweep
- N=2: λ=0.01 optimal (MSE -18%)
- N=10: λ=0.5 optimal (MSE -37%)
- **Rule**: Binding strength scales with object count

### 2. Graph Density Experiment  
- Density=0%: MLP ≈ GNN
- Density≥0.3: GNN always wins (max 9% gap)
- **Rule**: Graph Density ≥0.3 is threshold for relational reasoning

### 3. Unified Phase Surface
- Low Binding + Low Density → MLP region
- High Binding + High Density → GNN region
- Critical point: λ ≈ 0.1, Density ≈ 0.3

## Key Formula

**Constraint Interaction Law:**
```
能力涌现阈值 = λ_Binding × Density_Graph > 0.03
```

## Files Generated
- fig4_phase_surface.png
- fig5_unified_phase.png
- binding_sweep_results.json
