# FCRS-MIS 最小实验系统

> 500行代码，一台普通电脑即可启动
> 验证「压缩约束是否导致结构形成」

## 系统架构

```
MovingDotEnv → FCRS-MIS Micro-Unit → Metrics & Visualization
     ↓                ↓                    ↓
 移动点序列      100单元局部连接       聚类/结构熵/相变
```

## 核心假设

**压缩约束 λ → 结构涌现**

- λ=0: 无压缩，随机行为
- λ临界值: 相变，发生结构涌现
- λ过大: 表征坍缩

## 运行方式

```bash
pip install numpy matplotlib scikit-learn
python fcrs_mis_minimal.py
```

## 预期结果

| λ | 聚类质量 | 结构熵 | 行为 |
|---|---------|--------|------|
| 0 | 低 | 高 | 随机 |
| 0.05-0.2 | **阶跃上升** | **阶跃下降** | **相变** |
| >0.5 | 低 | 低 | 表征坍缩 |

## 文件

- `fcrs_mis_minimal.py` - 完整代码
- `fcrs_mis_results.png` - 可视化结果
