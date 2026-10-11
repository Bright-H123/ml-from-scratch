# C2 W1 — 神经网络：直觉、模型、前向传播、向量化

对应代码：`mlkit/neural_net.py`

---

## 1. 从逻辑回归到神经网络

一个逻辑回归单元 `a = σ(w·x + b)` 就是**一个神经元**。神经网络无非是把很多这样的单元并联成"层"，再把层串起来。

以"T 恤会不会畅销"为例：
- 输入 x = [价格, 运费, 营销投入, 材质]
- 隐藏层 3 个神经元，分别（在训练后**自发地**）学会表示"负担得起"、"知名度"、"感知质量"
- 输出层 1 个神经元，把这三个"高级特征"合成一个概率

**关键点**：你不需要告诉网络"造一个叫负担得起的特征"。你只提供 x 和 y，隐藏层的含义是训练出来的。**这就是神经网络相对手工特征工程的核心优势——特征工程自动化了。**

## 2. 记号

```
a⁽ˡ⁾ⱼ  = 第 l 层第 j 个神经元的激活值
w⁽ˡ⁾ⱼ, b⁽ˡ⁾ⱼ = 该神经元的参数
a⁽⁰⁾ = x  （输入层记作第 0 层，不计入"层数"）
```

单个神经元：

```
a⁽ˡ⁾ⱼ = g( w⁽ˡ⁾ⱼ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾ⱼ )
```

**注意 `w⁽ˡ⁾ⱼ` 的输入是"上一层的整个激活向量"**，不是原始 x。这就是"层层递进"的意思。

"这个网络有几层"：数隐藏层 + 输出层，**不数输入层**。[4, 3, 1] 是两层网络。

## 3. 前向传播

```
x → a⁽¹⁾ → a⁽²⁾ → … → a⁽ᴸ⁾ = 输出
```

一层一层往前算，每层的输出是下一层的输入。典型架构是**神经元数逐层递减**（如 25 → 15 → 1）。

## 4. 矩阵形式（本仓库用的约定）

课程的 lab 和 Keras 都用 **一行一个样本**：

```
A_prev  (m, n_in)     m 个样本，每行 n_in 个激活值
W       (n_in, n_out) 一列对应一个神经元
b       (n_out,)
Z = A_prev @ W + b    (m, n_out)   b 靠广播加到每一行
A = g(Z)
```

`mlkit/neural_net.py` 的 `dense_forward` 就是这三行。为什么 W 是 (n_in, n_out) 而不是转置：这样 `A @ W` 直接成立，不需要转置，和 Keras 的 `layer.get_weights()[0]` 形状一致。

一个 Dense 层的参数量 = `n_in × n_out + n_out`。

## 5. TensorFlow / Keras 对照

课程用 Keras 写：

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=1,  activation='sigmoid'),
])
model.compile(loss=tf.keras.losses.BinaryCrossentropy())
model.fit(X, y, epochs=100)
p = model.predict(X_new)
```

本仓库等价写法（**不需要装 TensorFlow**）：

```python
from mlkit.neural_net import NeuralNetwork

model = NeuralNetwork([n_features, 25, 15, 1], ['relu', 'relu', 'sigmoid'])
model.fit(X, y, epochs=100, alpha=0.01, batch_size=32)
p = model.predict_proba(X_new)
```

差别只有一处：Keras 从数据里推断输入维度，这里要显式写在 `layer_sizes[0]`。

### TF 的两个坑

1. **数据形状**：TF 要 `X` 是 2-D 的 `(m, n)`，一行一个样本。`np.array([200, 17])` 是一维的，必须写成 `np.array([[200, 17]])`。
2. **`model.predict` 返回的是 numpy 数组**，不是标量。

## 6. 向量化为什么重要

标量循环版本：

```python
for j in range(units):            # 每个神经元
    z = np.dot(W[:, j], a_in) + b[j]
    a_out[j] = g(z)
```

向量化版本：

```python
A_out = g(A_in @ W + b)           # 一次算完所有样本 × 所有神经元
```

底层是 BLAS 的 GEMM（通用矩阵乘），会用 SIMD 指令 + 多核 + 缓存分块。GPU 上更是几千个核同时算。**深度学习这十年能起来，很大程度上就是因为矩阵乘在硬件上极其高效。** 写 for 循环等于放弃这一切。

## 7. 关于 AGI

课程有一节讲这个，态度值得记住：
- ANI（狭义 AI，做特定任务）确实进步巨大
- AGI（通用 AI）没有明显进展，也没人知道要多久
- "神经网络模拟大脑"这个类比在今天已经**基本不成立**了——人工神经元和真实神经元的差距太大。别拿它当技术论据。

---

## 8. 常见坑

| 坑 | 现象 |
|---|---|
| 所有权重初始化为 0 | 同层所有神经元算出完全相同的东西，永远打不破对称，等价于只有 1 个神经元 |
| 忘了缩放输入 | 收敛极慢或不收敛 |
| 隐藏层用 sigmoid | 梯度消失，训练慢 |
| 输入形状写成一维 | Keras 报维度错，或者静默按错误方向广播 |

代码里 `Dense.__init__` 用的是 **He 初始化**（ReLU 层 `scale=√(2/n_in)`）/ Xavier（其他层 `√(1/n_in)`），随机且方差合适，既打破对称又不至于一开始就饱和。
