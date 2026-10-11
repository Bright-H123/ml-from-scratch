# C2 W2 — 训练神经网络、激活函数、Softmax、Adam、反向传播

对应代码：`mlkit/neural_net.py`

---

## 1. 训练模型的三步（对任何模型都成立）

| 步骤 | 逻辑回归 | 神经网络 (Keras) |
|---|---|---|
| ① 定义模型 f(x) | `z = w·x+b; f = 1/(1+e⁻ᶻ)` | `Sequential([Dense(...), ...])` |
| ② 定义损失和代价 | `L = −y log f −(1−y)log(1−f)` | `model.compile(loss=BinaryCrossentropy())` |
| ③ 最小化 J | 梯度下降 | `model.fit(X, y, epochs=100)` |

**术语区分**：loss 是单个样本的，cost 是整个训练集上 loss 的平均。

## 2. 激活函数怎么选

### 输出层——由 y 的类型决定，没有选择余地

| y 是什么 | 输出层激活 | 损失 |
|---|---|---|
| 二分类 0/1 | `sigmoid` | BinaryCrossentropy |
| 多分类 | `softmax` | SparseCategoricalCrossentropy |
| 回归，可正可负 | `linear` | MeanSquaredError |
| 回归，只能非负（如房价） | `relu` | MeanSquaredError |

### 隐藏层——默认 ReLU

`g(z) = max(0, z)`

为什么不用 sigmoid：
1. **sigmoid 两端都平**，z 大或小时梯度都≈0，梯度下降几乎不动（梯度消失）。ReLU 只有左边平。
2. ReLU 计算便宜（一次比较），sigmoid 要算指数。

为什么不能用 linear：

```
a⁽¹⁾ = W₁x + b₁
a⁽²⁾ = W₂a⁽¹⁾ + b₂ = W₂W₁x + (W₂b₁ + b₂) = W'x + b'
```

**全线性层堆叠等价于一个线性层**，网络深度白给。这也是为什么隐藏层绝不能用 linear。

## 3. Softmax 多分类

```
z_k = w_k·x + b_k,   k = 1..N
a_k = e^(z_k) / Σⱼ e^(z_j)
```

性质：所有 a_k > 0，且 **Σ a_k = 1**（所以能当概率分布）。N=2 时 softmax 化简回逻辑回归。

损失（交叉熵）：

```
L = −log(a_y)          只看真实类别对应的那个概率
```

只有一项是因为 one-hot 标签把其他项乘 0 了。所以 Keras 里叫 `SparseCategoricalCrossentropy`——"sparse" 指标签是整数 0/1/2 而非 one-hot 向量。

### ★ 数值稳定与 `from_logits=True`

朴素写法先算 `a = softmax(z)`，再算 `log(a)`。中间的 `e^z` 在 z 大时溢出，z 很负时下溢到 0 再取 log 就是 -inf。

正确做法是把 softmax 和 log 合并成一个表达式，中间不显式生成概率：

```python
# 课程推荐写法
model = Sequential([..., Dense(units=10, activation='linear')])   # 输出层用 linear！
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
# 预测时自己补 softmax
p = tf.nn.softmax(model(X))
```

本仓库的做法：`softmax()` 里先减去每行最大值再取指数（数学上完全等价，因为常数在分子分母约掉），`_output_dZ` 直接给出 `A − onehot(y)` 而不经过 log。实测 `softmax([1000, 1001, 999])` 正常返回 `[0.245, 0.665, 0.090]`，不溢出。

## 4. Adam 优化器

梯度下降用一个固定 α 走所有参数。Adam（**Ada**ptive **M**oment estimation）给**每个参数一个自己的学习率**：

- 某个参数总往同一方向挪 → 说明步子太小 → 加速
- 某个参数来回震荡 → 说明步子太大 → 减速

```
m = β₁m + (1−β₁)·g          一阶矩：梯度的滑动平均（方向）
v = β₂v + (1−β₂)·g²         二阶矩：梯度平方的滑动平均（幅度）
m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ)   偏差校正（m,v 从 0 开始，前期偏小）
θ := θ − α·m̂/(√v̂ + ε)
```

默认 `β₁=0.9, β₂=0.999, ε=1e-8`，α 初值 1e-3 左右。**Adam 对 α 的选择远比普通梯度下降宽容**，所以现在基本是默认选项。

Keras：`model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), ...)`
本仓库：`NeuralNetwork.fit()` 里已经内置 Adam，`alpha` 参数就是它的学习率。

## 5. 反向传播

### 为什么需要它

网络有几万个参数，如果每个都用有限差分算导数，每算一个就要跑一遍完整前向传播 → 总成本 O(参数量 × 前向成本)。反向传播用链式法则，**一次反向遍历就同时得到所有参数的导数**，成本约等于一次前向传播。这是让深度学习可行的关键算法。

### 推导

设 `Z⁽ˡ⁾ = A⁽ˡ⁻¹⁾W⁽ˡ⁾ + b⁽ˡ⁾`，`A⁽ˡ⁾ = g(Z⁽ˡ⁾)`。记 `δ⁽ˡ⁾ = ∂J/∂Z⁽ˡ⁾`。

**输出层**（三种配对都一样）：

```
δ⁽ᴸ⁾ = A⁽ᴸ⁾ − Y
```

sigmoid+BCE、softmax+CE、linear+MSE 三种组合都化简成"预测 − 真值"，原因和 C1W3 里说的一样：激活函数的导数恰好约掉了损失函数导数的分母。这个统一性不是巧合，是**指数族分布 + 对应的规范链接函数**的必然结果。

**逐层往回**：

```
∂J/∂W⁽ˡ⁾ = A⁽ˡ⁻¹⁾ᵀ δ⁽ˡ⁾ / m
∂J/∂b⁽ˡ⁾ = Σ_rows δ⁽ˡ⁾ / m
δ⁽ˡ⁻¹⁾   = (δ⁽ˡ⁾ W⁽ˡ⁾ᵀ) ⊙ g′(Z⁽ˡ⁻¹⁾)
```

`⊙` 是逐元素乘。各激活的导数：

```
ReLU'(z)    = 1 if z > 0 else 0
sigmoid'(z) = a(1−a)         （用已算好的 a，不用重算）
linear'(z)  = 1
```

代码里 `Dense.backward(dZ)` 做前两行，`_activation_backward` 做第三行的 `⊙ g′` 部分。

### ★ 数值梯度检验

写完反向传播**必须**验：

```python
net = NeuralNetwork([4, 5, 3, 1], ['relu', 'relu', 'sigmoid'], lambda_=1.3)
print(net.gradient_check(X, y))     # 实测 1.5e-10
```

判据：差 < 1e-7 基本正确；> 1e-5 一定有 bug。本仓库对 sigmoid/softmax/linear 三种输出、带与不带 L2，全部通过，最差 3.0e-10。

## 6. 其它要点

- **多标签分类 (multi-label)** ≠ 多分类 (multi-class)。多标签是"图里有没有车/有没有公交/有没有行人"，三个独立的是/否 → 输出层 3 个 **sigmoid** 单元。多分类是"这是 0-9 中哪个数字" → 一个 **softmax** 层。
- **Advanced optimization**：Adam 之外还有很多，但课程只要求会用 Adam。
- **卷积层 (Convolutional layer)**：每个神经元只看输入的一个局部窗口。好处是参数更少、需要的训练数据更少、更不容易过拟合。课程只做概念介绍。

---

## 7. 常见坑

| 坑 | 现象 | 修法 |
|---|---|---|
| 输出层用 softmax 又用 `from_logits=True` | 相当于做了两次 softmax，结果错但不报错 | 输出层设成 `linear` |
| 忘了在 predict 时补 softmax | 输出是 logits 不是概率，可能是负数 | `tf.nn.softmax(model(X))` |
| 反向传播写完不做梯度检验 | 训练"能跑"但学不好，极难 debug | `gradient_check()` |
| 学习率过大 | cost 变 NaN | 减小 alpha；用 Adam |
