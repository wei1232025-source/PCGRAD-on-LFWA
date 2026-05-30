# LFWA 梯度手术实验设计

## 1. 实验目标

课程要求是实现梯度手术算法，并验证它在处理多目标冲突时的优越性。因此本实验不追求证明“彻底消除性别信息”，而是构造一个真实数据上的冲突任务：

- 目标 1（效用）：利用共享 Backbone 特征高精度预测人脸是否在微笑（Smiling）。
- 目标 2（公平/去信息）：让共享 Backbone 特征尽量不包含可被性别攻击器利用的 Male 信息。

这两个目标天然可能冲突：如果 LFWA 中 Smiling 与 Male 存在统计相关性，模型为了提高微笑预测精度，可能会把性别作为捷径特征；而公平目标又要求 Backbone 去除这类性别信息。

## 2. 模型结构

代码采用三部分结构：

```text
Image -> Shared Backbone -> Feature
Feature -> Smile Head -> Smiling prediction
Feature -> Gender Head -> Male prediction
```

其中：

- `Backbone` 是共享特征提取器。
- `Smile Head` 学习效用任务，即预测 Smiling。
- `Gender Head` 是攻击器，用于检测 Backbone 特征中是否仍然保留性别信息。

训练时，Gender Head 自己正常最小化性别分类损失；Backbone 则接收相反方向的性别梯度，试图让 Gender Head 无法成功预测性别。

## 3. 多目标冲突形式

Backbone 面临两个梯度：

```text
g_smile   = gradient(L_smile)
g_fair    = gradient(-lambda * L_gender)
```

其中：

- `L_smile` 越小，微笑分类越好。
- `L_gender` 越小，性别攻击器越强。
- Backbone 使用 `-L_gender`，表示它要破坏性别攻击器。

如果：

```text
g_smile · g_fair < 0
```

说明两个目标在当前 batch 上发生梯度冲突。普通加权法会直接相加，可能导致某个目标被破坏；梯度手术则会处理冲突方向。

## 4. 实现的梯度手术算法：PCGrad

本实验实现 PCGrad。核心思想是：当两个任务梯度夹角为负时，去掉一个梯度在另一个梯度反方向上的冲突分量。

两任务情况下，代码中使用：

```text
if g1 · g2 < 0:
    g1 <- g1 - (g1 · g2 / ||g2||^2) g2
    g2 <- g2 - (g1 · g2 / ||g1||^2) g1

g_final = g1 + g2
```

这样 Backbone 的更新方向会尽量同时服务于微笑预测和去性别信息，而不是简单让两个目标互相抵消。

## 5. 对比实验

代码默认运行三组实验：

```text
smile_only：只训练微笑预测，作为效用上限。
naive_adv：普通对抗加权，直接优化 L_smile - lambda * L_gender。
pcgrad：在 L_smile 与 -lambda * L_gender 的 Backbone 梯度之间做梯度手术。
```

期望现象：

- `smile_only` 的 Smiling 精度较高，但 Gender Head 可能仍能从特征中预测性别。
- `naive_adv` 能降低部分性别信息，但在强冲突时可能损害 Smiling 精度。
- `pcgrad` 应在相近 Smiling 精度下，使 Gender Balanced Accuracy 更接近 50%，表现出更好的冲突处理能力。

## 6. 评价指标

实验输出以下指标：

- `Smile Acc`：Smiling 分类准确率，越高越好。
- `Gender Balanced Accuracy`：性别攻击器的平衡准确率，越接近 50% 越好。
- `Gradient Cosine Similarity`：两个目标在 Backbone 上的梯度余弦相似度；小于 0 表示发生冲突。
- `Conflict Rate`：训练中发生负梯度内积的 batch 比例。

这里不建议只看性别普通准确率，因为 LFWA 中 Male/Female 样本比例可能不均衡。若男性比例较高，攻击器即使简单预测多数类，也可能得到较高普通准确率。因此代码使用 Balanced Accuracy 作为主要公平指标。

## 7. 数据比例影响

男性图片比例会影响实验效果，主要体现在：

1. 如果 Male/Female 不均衡，性别攻击器的普通 accuracy 会偏向多数类，指标不可靠。
2. 如果 Smiling 与 Male 标签存在相关性，模型会倾向于利用性别作为预测 Smiling 的捷径特征，使两个目标冲突更明显。
3. Backbone 接收到的公平梯度可能受多数性别样本影响，因此需要报告 Male/Female 与 Smiling 的四格统计。

代码启动时会输出：

```text
Male & Smiling
Male & Not Smiling
Female & Smiling
Female & Not Smiling
```

这可以用于解释数据集中的标签相关性，以及为什么该任务确实构成多目标冲突。

## 8. 数据格式

默认路径为：

```text
data/lfwa/attributes.csv
data/lfwa/images/
```

CSV 至少需要包含三列：

```text
image,Smiling,Male
```

示例：

```csv
image,Smiling,Male
Aaron_Eckhart_0001.jpg,1,1
Abbie_Cornish_0001.jpg,0,0
```

代码也支持 CelebA/LFWA 属性 txt 风格文件：第一行为样本数，第二行为属性名，后续每行为图片名和属性值。属性值可以是 `-1/1` 或 `0/1`。

## 9. 运行方式

```bash
python main.py --data-root data/lfwa --annotation-file data/lfwa/attributes.csv
```

常用参数：

```bash
python main.py --epochs 20 --batch-size 64 --adv-weight 0.2
```

实验结束后会生成：

```text
lfwa_pcgrad_comparison.png
```

图中包含 Smiling 精度、性别攻击平衡准确率和梯度冲突余弦曲线。

## 10. 结论表述建议

报告中建议使用稳妥表述：

> 在相同或接近的 Smiling 预测精度下，PCGrad 相比普通对抗加权能使 Gender Balanced Accuracy 更接近随机水平，同时减少负梯度冲突造成的训练干扰，说明梯度手术在多目标冲突优化中具有更好的折中能力。

不建议表述为“彻底遗忘性别”或“任何攻击者都无法预测性别”，因为实验只能证明在当前攻击器和当前数据划分下，性别信息泄露被显著降低。
