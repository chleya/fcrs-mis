# Variable Emergence Phase Diagram

**Research Question**: How do variables emerge in representations? Is meta-structure necessary for intelligence?

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

## Core Formula

```
Variable Emergence = f(Structure, Data, Excitation)

I(z; v) > I(z; a)  ← Identifiability Condition
```

## Three-Region Phase Diagram

1. **Shortcut Zone**: Low structure + High data → Corr(z,a) > Corr(z,v)
2. **Statistical Recovery**: High data + Weak structure → |Corr| ≈ 0.5
3. **Causal Recovery**: Causal structure + Medium excitation → |Corr| ≈ 0.99

## Experiments

```bash
# Core experiment
python mini_experiment.py

# Single-variable verification
python single_variable_experiment.py

# Numerical stability verification
python stability_test.py

# Ablation experiment
python ablation_experiment.py

# Excitation sweep
python excitation_sweep.py

# Multi-environment verification
python multi_env_experiment.py
```

## Test Files

| File | Description |
|------|-------------|
| `mini_experiment.py` | Core comparison (Baseline vs Causal) |
| `single_variable_experiment.py` | Single-variable control |
| `stability_test.py` | Numerical stability verification |
| `ablation_experiment.py` | Component necessity verification |
| `excitation_sweep.py` | Excitation intensity sweep |
| `multi_env_experiment.py` | Multi-environment generalization |

## Key Discoveries

1. **Encoder Isolation is Critical**: Removing action from encoder causes -1.159 correlation change
2. **Dynamics and Decoder are NOT Critical**: Minor changes when removed
3. **Excitation Affects Emergence**: Higher force generally improves correlation
4. **Generalization**: Works on CartPole and SpringMass, but not Pendulum

## Mathematical Formalization

See: `memory/mathematical_formalization.md`

---

*Research Area: Representation Learning / Causal Representation*
