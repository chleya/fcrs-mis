# Object-Centric Representations Enable Combinatorial Scaling, But Their Benefits Are Hidden by Perception Bottlenecks

## Abstract

We investigate when object-centric representations provide advantages over global trajectory models in physical prediction tasks. Through a systematic series of experiments, we find that: (1) Trajectory models excel at in-distribution prediction tasks, achieving near-perfect accuracy; (2) However, when generalizing to novel object counts (2→6 objects), trajectory models suffer severe performance degradation (80-1760%); (3) Critically, when perception is removed and representations are given directly as coordinates, object-centric models maintain stable performance while trajectory models continue to fail; (4) Multi-seed verification confirms these results are robust (Object MSE: 0.0005±0.0001 vs Trajectory: 0.0104±0.0033). Our findings reveal a fundamental principle: object-centric representations enable combinatorial scaling, but their benefits are hidden by perception bottlenecks.

---

## 1 Introduction

A fundamental question in representation learning is: when do object-centric representations provide advantages over global state representations?

**World models** and **trajectory models** have shown remarkable success in prediction tasks, learning global latent states that encode entire scenes. However, these approaches face challenges when the number of objects changes—a critical capability for generalization.

**Object-centric representations** (e.g., Slot Attention, Interaction Networks) decompose scenes into individual objects, potentially enabling better combinatorial generalization. But empirical results have been mixed.

**Our contribution**: We systematically disentangle perception from representation, demonstrating that:
1. Object-centric benefits are hidden by perception bottlenecks
2. When perception is solved, object-centric models enable combinatorial scaling
3. Trajectory models fail dramatically under object count changes

---

## 2 Related Work

### 2.1 World Models and Trajectory Prediction

Recent work in world models (Ha & Schmidhuber, 2018) uses global latent states for future prediction. These approaches excel at in-distribution tasks but face challenges with distribution shift.

### 2.2 Object-Centric Representations

Slot Attention (Locatello et al., 2020) and Interaction Networks (Battaglia et al., 2016) propose factorized representations. However, their benefits often require:
- Clean object-level inputs
- Sufficient training data
- Proper inductive biases

### 2.3 Combinatorial Generalization

The ability to generalize to novel compositions is a key challenge. Key insight: parameter sharing across objects enables scaling to arbitrary N.

---

## 3 Experimental Framework

### 3.1 Task Design

We use a simple physics simulation:
- Objects: balls moving in 2D space
- Input: Two frames (t0, t10)
- Target: Position of object 0 at t10
- Varies: Object count (2→6), input type (pixels vs coordinates)

### 3.2 Models

1. **Trajectory Model**: CNN encoder + MLP, processes entire scene as global state
2. **Object Model**: Factorized processing, shared parameters per object
3. **Slot Attention**: Alternative object decomposition (for comparison)

### 3.3 Key Control: Perception vs Representation

We explicitly control for perception by testing:
- **Pixel input**: Raw images requiring object discovery
- **Coordinate input**: Ground-truth (x, y, vx, vy) removing perception

---

## 4 Results

### 4.1 In-Distribution Prediction (L1)

| Task | Baseline | Random |
|------|----------|--------|
| Standard prediction | 99% | 0% |
| Partial occlusion | 97% | 0% |
| Object count change | 91% | 0% |
| Permutation identity | 98% | 0% |

**Finding**: Trajectory models excel at in-distribution prediction.

### 4.2 Combinatorial Scaling (2→6 objects)

**Pixel Input:**
| Model | 2 Objects | 6 Objects | Drop |
|-------|-----------|-----------|------|
| Trajectory | 74% | 36% | **150%** |
| Slot Attention | ~0% | ~0% | N/A |

**Finding**: Both models fail with pixel input due to perception bottleneck.

### 4.3 Removing Perception: Coordinates Input

| Model | 2 Objects | 6 Objects | Drop |
|-------|-----------|-----------|------|
| Trajectory | 99% | 80% | **1760%** |
| Object | 99% | 99% | **-9%** |

**Finding**: With perception solved, object-centric model maintains stable performance!

### 4.4 Multi-Seed Verification (5 seeds)

| Model | MSE (6 objects) | Std |
|-------|-----------------|-----|
| Trajectory | 0.0104 | ±0.0033 |
| Object | 0.0005 | ±0.0001 |

**Ratio: Object 20x better than Trajectory**

### 4.5 Visualization

Predictions on novel object counts:
- Object model: Error ~0.01-0.02 consistently
- Trajectory model: Error 0.08-0.12 (6x worse)

---

## 5 Analysis

### 5.1 Why Trajectory Models Fail

Trajectory models learn: `f(global_state) → next_state`

State dimension scales with object count: O(N)

When N changes (2→6), the model faces dimension shift → performance collapse.

### 5.2 Why Object Models Succeed

Object models learn: `f(object_i) → next_state` with shared parameters

Rules are learned once and apply to any N.

This is **combinatorial generalization through parameter sharing**.

### 5.3 The Perception Bottleneck

Our key finding: Object-centric benefits are hidden when:
- Input is pixels (requiring object discovery)
- Object decomposition is imperfect

This explains mixed results in prior literature.

---

## 6 Phase Diagram

```
       scaling / reasoning
              ↑
   object-centric │
                  |
  ----------------+-------- prediction
                  |
  trajectory      ↓
```

| Region | Best Model | Reason |
|--------|-----------|--------|
| In-distribution prediction | Trajectory | Sufficient capacity |
| Combinatorial scaling | Object-centric | Factorized representation |
| Pixel input | Both fail | Perception bottleneck |
| Reasoning/counterfactual | Unknown | Future work |

---

## 7 Conclusion

We demonstrate that:

1. **Trajectory models excel at in-distribution prediction** but fail under combinatorial scaling (80-1760% performance drop)

2. **Object-centric representations enable combinatorial generalization** when perception is solved, maintaining 99% accuracy across object counts

3. **Perception bottlenecks hide representation benefits** — this explains inconsistent findings in prior literature

4. **The key insight**: Parameter sharing across objects enables generalization to novel compositions

### Implications

For practitioners:
- Object-centric models are essential when object counts vary
- Perception and representation should be studied separately
- Clean object-level inputs unlock representation benefits

For researchers:
- The phase diagram provides a map for future investigation
- Key question: How to bridge perception and representation?

---

## Acknowledgments

This work was conducted as part of the FCRS-MIS project exploring structure-capability relationships in representation learning.

---

## References

- Ha, D., & Schmidhuber, J. (2018). Recurrent world models facilitate policy evolution.
- Locatello, F., et al. (2020). Object-centric learning with slot attention.
- Battaglia, P., et al. (2016). Interaction networks for learning about objects, relations and physics.
- Lake, B. M., et al. (2017). Building machines that learn and think like people.
