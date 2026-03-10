# When Do Predictive Systems Learn Dynamical Variables?

## An Empirical Study on Goal-Driven Representation Emergence

---

## Abstract

We investigate the conditions under which predictive learning systems spontaneously encode dynamical variables (such as velocity) in their internal representations. Through a series of controlled experiments in a minimal 2D agent environment, we find that: (1) pure prediction learning primarily encodes position information; (2) survival-driven goal increases velocity-related information in representations; (3) however, ablation experiments reveal that velocity encoding is not causally necessary for survival - the system can achieve survival using alternative strategies. We conclude that goal-driven interaction biases representations toward dynamical variables, but does not necessarily produce causally necessary state variables.

---

## 1. Introduction

### 1.1 Background
- Predictive learning is a fundamental paradigm in AI (World Models, Predictive Coding)
- Key question: Do predictive systems automatically discover physical state variables?
- Common assumption: Prediction → Causal representations

### 1.2 Our Approach
- Minimal 2D moving point environment
- Systematic ablation of three key factors:
  - Prediction horizon
  - Compression constraint
  - Goal structure (survival vs prediction)

---

## 2. Minimal Predictive System

### 2.1 Environment
- 2D agent moving in bounded space
- Observation: current position (x, y)
- Target: predict future position
- Dynamics: x(t+1) = x(t) + v(t) + noise

### 2.2 Model Architecture
- Simple MLP: input → hidden → output
- Compression constraint: L1 regularization on weights
- Training: minimize prediction MSE + compression loss

---

## 3. Prediction-Only Experiments

### 3.1 Experiment 1: Prediction Horizon Sweep

| Horizon T | MI(velocity) | MI(position) |
|-----------|--------------|--------------|
| 1 | 0.04 | 0.76 |
| 50 | 0.07 | 0.01 |

**Finding**: Longer prediction horizon shifts representation from position toward velocity, but neither becomes dominant.

### 3.2 Experiment 2: Compression Constraint

| λ | Silhouette | Structure |
|---|------------|-----------|
| 0 | 0.46 | weak |
| 0.01 | 0.90 | strong |

**Finding**: Compression induces structural organization in representations.

### 3.3 Experiment 3: Linear Probe Analysis

| Target | R² |
|--------|-----|
| velocity | 0.12 |
| position | 0.23 |

**Finding**: Pure prediction encodes position more than velocity; no factorization into state variables.

---

## 4. Action Experiments

### 4.1 Experiment 4: Passive vs Active

| Condition | R²(velocity) |
|-----------|--------------|
| Passive (no action) | 0.15 |
| Fixed action | 0.16 |
| Learned action | 0.15 |

**Finding**: Action alone, without survival goal, does not increase velocity encoding.

---

## 5. Survival Experiments

### 5.1 Experiment 5: Survival-Driven Learning

Environment: Energy-based survival task
- Agent must track and reach energy source
- Action costs energy
- Energy depletion = death

| Model | R²(velocity) | Variance |
|-------|--------------|----------|
| Prediction | 0.10 | 0.02 |
| Survival | 0.82 | 0.001 |

**Finding**: Survival goal dramatically increases velocity encoding (+717%).

### 5.2 Experiment 6: Strict Environment (Single Frame Input)

To eliminate temporal shortcuts, we use single-frame observation:

| Model | R²(velocity) |
|-------|--------------|
| Prediction | 0.04 |
| Survival | 0.26 |

**Finding**: Even with minimal observation, survival increases velocity information.

### 5.3 Experiment 7: Cross-Environment Invariance

Test with fixed probe trained on α=1.0, tested on α=0.5/2.0:

| Model | α=0.5 | α=1.0 | α=2.0 |
|-------|-------|-------|-------|
| Prediction | negative | negative | negative |
| Survival | 0.12 | 0.23 | 0.43 |

**Finding**: Survival-learned representations generalize across environments; prediction representations collapse.

---

## 6. Representation Analysis

### 6.1 Experiment 8: Ablation Test

Question: Is velocity encoding causally necessary for survival?

Method: Full velocity subspace projection ablation

| Condition | Survival Steps |
|-----------|----------------|
| Baseline | 14.8 |
| Velocity ablated | 14.5 |

**Finding**: Velocity encoding is NOT necessary for survival. System uses alternative strategies (position tracking, distance-based control).

### 6.2 Interpretation

The system can achieve survival through:
- Relative position tracking
- Distance-based control
- Proportional control to energy source

Velocity information increases but is not causally required.

---

## 7. Discussion

### 7.1 What We Found

| Condition | Result |
|-----------|--------|
| Prediction only | Position dominant |
| Survival goal | Velocity information increases |
| Ablation | Velocity not necessary |

**Core phenomenon**: Goal-driven learning biases representations toward dynamical variables, but does not guarantee causal necessity.

### 7.2 Theoretical Implications

1. **Prediction ≠ Causation**: Pure predictive learning learns statistical patterns, not causal structures

2. **Goal Biases Representation**: Survival goal pushes system toward dynamical variables, but flexible

3. **Distributed Encoding**: Velocity information may be encoded distributively, making targeted ablation ineffective

### 7.3 Limitations

- Minimal environment may not capture full complexity
- Ablation method may not capture all velocity information
- Further work needed on causal necessity

---

## 8. Conclusion

We investigated when predictive systems learn dynamical variables through systematic experiments in a minimal 2D environment. Our key findings:

1. **Pure prediction** primarily encodes position information, not velocity
2. **Survival goal** significantly increases velocity-related information in representations (+717%)
3. **Cross-environment** survival representations generalize; prediction representations collapse
4. **Causal necessity** is not confirmed: ablation does not reduce survival performance

We conclude that **goal-driven interaction biases representations toward dynamical variables, but does not necessarily produce causally necessary state variables**. This empirical result provides a clear boundary for understanding when predictive systems may learn physical causality.

---

## References

- World Models (Ha & Schmidhuber, 2018)
- Predictive Coding (Rao & Ballard, 1999)
- Information Bottleneck (Tishby et al., 2000)
- Invariant Risk Minimization (Arjovsky et al., 2019)

---

## Appendix: Experiment Parameters

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 32 |
| Learning rate | 0.01 |
| Compression λ | 0.01 |
| Training steps | 3000 |
| Seeds | 5 |
| Environment | 2D bounded space |

---

*Generated: 2026-03-11*
*Project: FCRS-MIS (Folding-Constrained Representation System)*
