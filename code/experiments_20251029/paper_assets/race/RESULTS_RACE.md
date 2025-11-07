# RACE 阶段结果小结（三模型对比）

**文件索引（每模型一组）：**
- `overall.csv`：总体准确率 / 置信度 / ECE / Brier / 稳定性
- `by_human.csv`：按人类难度（middle / high）分组指标
- `datamap_counts.csv`：Data Map 四象限样本量（Easy / Ambiguous / Hard / Impossible）
- `datamap.png`：Data Map 可视化
- `reliability.png`：可靠性校准曲线（ECE）

**撰写要点建议：**
1) 三模型总体准确率均高；更强模型在 `datamap_counts.csv` 中 Easy 占比更高、Ambiguous 更低。  
2) `by_human.csv` 中，所有模型在 middle > high 的准确率趋势一致，体现与人类难度方向一致。  
3) `reliability.png` 对比 ECE 与高置信区表现，指出过度自信或更佳校准的模型。  
4) 用 `datamap.png` 说明“模糊区”体量与分布差异，联动稳定性统计。

> 数值请直接引用对应 CSV，避免手抄误差。
