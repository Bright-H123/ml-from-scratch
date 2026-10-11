# C2 W4 — 决策树、信息增益、随机森林、XGBoost

对应代码：`mlkit/decision_tree.py`、`mlkit/ensembles.py`

---

## 1. 决策树是什么

一棵二叉树，每个内部节点问一个关于某个特征的问题，叶子给出预测。

```
solitary <= 0.5?
    yes -> tapering_stalk <= 0.5?
        yes -> predict 0
        no  -> predict 1
    no  -> brown_cap <= 0.5?
        yes -> predict 0
        no  -> predict 1
```

（这是 `mlkit` 在课程的蘑菇数据上真实跑出来的树，根节点信息增益 0.2781，和课程讲义的 0.278 一致。）

**和课程其它所有算法最大的不同：树不是用梯度下降训练的。** 分裂结构是离散的，没有梯度可求。取而代之的是**贪心搜索**：在每个节点试遍所有 (特征, 阈值) 组合，选让不纯度下降最多的那个，然后递归。

贪心 = 每个节点局部最优，不保证整棵树全局最优。接受这一点是因为找最优树是 NP-hard，而贪心的效果实践上够好。

## 2. 熵：怎么衡量"不纯"

```
H(p₁) = −p₁log₂(p₁) − p₀log₂(p₀)
```

- p₁ = 0 或 1（节点纯）→ H = 0，没有不确定性
- p₁ = 0.5 → H = 1.0，最大不确定性（抛硬币）
- 曲线是一个开口向下、峰在 0.5 的钟形

约定 `0·log(0) = 0`（这是极限值）。

**基尼系数** `Gini = 1 − Σpₖ²` 效果几乎一样，计算更便宜（不用算 log）。sklearn 默认用 gini，课程讲 entropy。

## 3. 信息增益：怎么选分裂

```
IG = H(parent) − [ w_left·H(left) + w_right·H(right) ]
```

其中 `w_left = 左子节点样本数 / 父节点样本数`。

**★ 为什么要按样本数加权**：不加权的话，"分出一个纯的单样本节点"看起来和"干净地对半分"一样好。但大而杂的节点才是真正的代价所在。加权保证算法优先处理大节点。

选择规则：试遍所有特征、所有可能阈值，取 IG 最大的。

## 4. 停止条件（控制过拟合的旋钮）

课程列了四条：
1. 节点已经纯了（全是一类）
2. 达到最大深度 `max_depth`
3. 最好的 IG 也小于阈值 `min_impurity_decrease`
4. 节点样本数少于 `min_samples_split`

**每一条都是偏差-方差的旋钮**：树越深 → 偏差越低、方差越高。不设限制的树能在训练集上做到 100% 正确，但那是纯粹的过拟合。

选 `max_depth` 用 cv 集（C2W3 的方法在这里同样适用）。

## 5. 独热编码（One-hot）

特征有 k>2 个取值时（耳形 = 尖/垂/椭圆），造 k 个 0/1 特征，恰好一个是 1。

好处：树只需要处理二元分裂，代码简单。而且**同一套独热编码后的特征可以直接喂给神经网络和逻辑回归**——它们本来就要求数值输入。

k=2 时不用独热，直接编成一个 0/1 特征。

## 6. 连续特征

不能穷举所有实数阈值。做法：把该特征的值排序，取**相邻两个不同值的中点**作为候选阈值，逐个试。m 个样本最多 m−1 个候选。

`mlkit` 的 `best_split()` 就是这么实现的，二值/独热特征只是"候选阈值只有 0.5 一个"的特例，不需要单独写代码。

## 7. 回归树

只改两处：
- **不纯度用方差代替熵**
- **叶子预测该节点 y 的均值**

其余完全一样。结果是一个**分段常数函数**——所以单棵回归树画出来是阶梯状的，不平滑。

## 8. 树集成

### 为什么要集成

单棵树对数据**极度敏感**：改动一个训练样本就可能让根节点的分裂完全变掉，长出一棵截然不同的树。这是典型的高方差。

**平均很多棵树可以把这种敏感性抵消掉。** 集成本质上是一个方差削减操作。

### 有放回抽样（Bootstrap）

从 m 个样本里**有放回**地抽 m 次，得到一个新训练集（有重复、有遗漏）。

- 每个 bootstrap 集大约包含原始样本的 **63.2%**（= 1 − 1/e）的不同样本
- 剩下约 36.8% 叫 **out-of-bag (OOB)**，可以当免费的验证集

### 袋装决策树 (Bagged trees)

抽 B 个 bootstrap 集，各训一棵树，预测时投票。B 取 64~128 常见；**B 再大也不会过拟合，只是收益递减**（超过 100 通常没意义，还慢）。

### 随机森林 = 袋装 + 特征随机

袋装有个弱点：即使换了 bootstrap 集，各棵树还是倾向于选同一个强特征当根，**树之间高度相关**，平均的收益就小。

修法：**每次分裂时只允许考虑随机的 k 个特征**（通常 `k = √n`）。强行让树们意见不一致。

```python
from mlkit.ensembles import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=40, max_depth=6, max_features='sqrt').fit(X, y)
print(rf.oob_score(X, y))     # 免费的验证估计
```

（实测和 sklearn 的 RandomForest 在 moons 数据上准确率 0.9875 vs 0.99，OOB 0.955 vs 0.9525。）

### XGBoost / 梯度提升

和随机森林的根本区别：

| | 随机森林 | 提升 (Boosting) |
|---|---|---|
| 树之间 | **独立**训练，最后平均 | **串行**训练，每棵拟合前面的残差 |
| 关注点 | 所有样本平等 | 刻意盯着**做错的样本** |
| 过拟合风险 | 低（树越多越稳） | 较高（轮数太多会拟合噪声） |

Andrew 的类比是"**刻意练习 (deliberate practice)**"：练琴不是从头到尾重弹，而是专挑弹不好的那几小节反复练。

```
f₀ = ȳ                                    初始预测 = 均值
for i in 1..B:
    residual = y − f_{i-1}                当前还差多少
    树 i 拟合 residual
    f_i = f_{i-1} + learning_rate × 树i(x)
```

学习率（收缩系数）把每棵树的贡献缩小，让模型慢慢逼近，是防过拟合的主要手段。

XGBoost 的实际实现还加了：二阶梯度、叶子权重的正则化、列采样、缺失值处理、以及大量工程优化。课程只要求会调用：

```python
from xgboost import XGBClassifier
model = XGBClassifier()          # 分类
model = XGBRegressor()           # 回归
model.fit(X_train, y_train)
```

本仓库的 `GradientBoostingRegressor` 实现了核心循环（实测 R²=0.9687，与 sklearn 的 GradientBoostingRegressor **完全一致**）。

## 9. ★ 决策树 vs 神经网络

| | 决策树 / 树集成 | 神经网络 |
|---|---|---|
| **结构化（表格）数据** | ★ 首选 | 也能用 |
| **非结构化数据**（图像/音频/文本） | 不推荐 | ★ 首选 |
| **混合型数据** | 可以 | 可以 |
| 训练速度 | 快 | 慢 |
| 可解释性 | 小树很好读；大集成其实也不好读 | 差 |
| 迁移学习 | 不支持 | ★ 支持（这是巨大优势） |
| 多模型串联 | 难（不可微） | ★ 容易（端到端梯度下降） |

判断口诀：**数据能放进 Excel 表格 → 先试 XGBoost；数据是图片/声音/文字 → 上神经网络。**

"可解释性"这一条要小心：单棵浅树确实好读，但一个 100 棵树的集成并不比神经网络更容易解释。**别把"我用了树所以模型可解释"当成理所当然。**

---

## 10. 代码对照

```python
from mlkit import datasets, decision_tree as dt, ensembles as ens

X, y, names = datasets.mushroom_data()
tree = dt.DecisionTreeClassifier(max_depth=2, criterion='entropy').fit(X, y)
print(tree.print_tree(names))          # 树能打印出来看，这是它最好的一点

rf = ens.RandomForestClassifier(n_estimators=50, max_features='sqrt').fit(X, y)
gb = ens.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
```
