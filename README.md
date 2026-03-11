# Variable Emergence Phase Diagram

**研究问题**: 变量如何在表示中涌现？元结构是否是智能产生的必要条件？

## 核心发现

### 因果架构 vs Baseline

| 模型 | Corr(z, angle) | 结论 |
|------|-----------------|------|
| Baseline | **-0.82** | 走action捷径 |
| Causal | **+0.89** | 学到真实变量 |

### 捷径检测 (MSE恶化倍数)

| 模型 | 恶化倍数 |
|------|----------|
| Baseline | **211x** |
| Causal | **103x** |

## 核心公式

```
变量涌现 = f(结构, 数据, 激励)

I(z; v) > I(z; a)  ← 可辨识性条件
```

## 三区域相图

1. **捷径区**: 低结构 + 高数据 → Corr(z,a) > Corr(z,v)
2. **统计恢复区**: 高数据 + 弱结构 → |Corr| ≈ 0.5
3. **因果恢复区**: 因果架构 + 中等激励 → |Corr| ≈ 0.99

## 实验

```bash
# 运行核心实验
python mini_experiment.py
```

## 关键文件

- `mini_experiment.py` - 极简对比实验
- `single_variable_experiment.py` - 单一变量验证

## 论文

详见: `memory/theoretical_phase_diagram.md`

---

*研究领域: Representation Learning / Causal Representation*
