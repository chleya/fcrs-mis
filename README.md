# Variable Emergence Phase Diagram

**Research Question**: How do variables emerge in representations? Is meta-structure necessary for intelligence?

## Abstract

We investigate whether causal structure (meta-structure) is necessary for intelligent representation learning. Through rigorous experiments on multiple physical environments, we demonstrate that:

1. **Causal architecture enables variable emergence**: Correlation with true variables improves from -0.82 to +0.89
2. **Encoder isolation is critical**: Ablation shows this component is necessary
3. **Theory generalizes**: 2/3 environments validate the framework

## Core Findings

### Causal Architecture vs Baseline

| Model | Corr(z, angle) | Conclusion |
|-------|----------------|------------|
| Baseline | **-0.82** | Takes action shortcut |
| Causal | **+0.89** | Learns true variable |

### Shortcut Detection (MSE Degradation)

| Model | Degradation |
|-------|-------------|
| Baseline | **211x** |
| Causal | **103x** |

### Ablation Analysis

| Component Removed | Change in Corr | Critical? |
|-----------------|----------------|------------|
| Encoder Isolation | **-1.16** | **YES** |
| Dynamics | +0.06 | NO |
| Decoder | +0.04 | NO |

## Theoretical Framework

### Three-Region Phase Diagram

```
        Excitation
             ↑
    Causal    |    Unstable
    Recovery  |    (high force)
             |
-------------+------------- Baseline
    Shortcut  |    Statistical
    (no struct)|    Recovery
             |
          Data →
```

### Mathematical Formalization

**Identifiability Condition**:
```
I(z; v) > I(z; a)
```

**Emergence Condition**:
```
E = S × D × I > τ
```

Where:
- S = Structural strength (0=MLP, 1=causal)
- D = Data coverage
- I = Excitation intensity
- τ = Critical threshold

## Experiments

### 1. Core Comparison (mini_experiment.py)
- Baseline vs Causal architecture
- Result: +0.89 correlation with true variables

### 2. Single-Variable Control (single_variable_experiment.py)
- Three-model controlled comparison
- Result: Causal architecture enables variable emergence

### 3. Numerical Stability (stability_test.py)
- Correlation computation verification
- Gradient flow verification
- Result: All tests passed

### 4. Ablation Analysis (ablation_experiment.py)
- Component necessity verification
- Result: Encoder isolation is critical

### 5. Excitation Sweep (excitation_sweep.py)
- Force intensity vs variable emergence
- Result: Higher force generally improves correlation

### 6. Multi-Environment (multi_env_experiment.py)
- CartPole, Pendulum, SpringMass
- Result: 2/3 environments validate theory

## Key Contributions

1. **Quantified meta-structure effect**: From -0.82 to +0.89 correlation improvement
2. **Identified critical component**: Encoder isolation prevents shortcuts
3. **Established phase diagram**: Three-region framework for variable emergence
4. **Mathematical formalization**: I(z; v) > I(z; a) identifiability condition

## Limitations

1. Pendulum environment shows anomalous results (action effect too small)
2. Theory needs more environment validation
3. Mathematical proofs are preliminary

## Citation

If this work helps your research, please cite:

```bibtex
@misc{fcrs-mis-2026,
  title={Variable Emergence Phase Diagram},
  author={Chen Leiyang},
  year={2026},
  url={https://github.com/chleya/fcrs-mis}
}
```

---

*Research Area: Representation Learning / Causal Discovery*
