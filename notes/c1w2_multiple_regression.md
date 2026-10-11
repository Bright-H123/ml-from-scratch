# C1 W2 — 多元线性回归、向量化、特征缩放、特征工程

对应代码：`mlkit/linear_regression.py`、`mlkit/preprocessing.py`

---

## 1. 多元线性回归

n 个特征时：

```
f(x) = w₁x₁ + w₂x₂ + … + wₙxₙ + b = w·x + b
```

记号约定（课程里很容易混）：
- `x⁽ⁱ⁾` 是**第 i 个样本**，是一个长度 n 的向量（行）
- `xⱼ⁽ⁱ⁾` 是第 i 个样本的第 j 个特征，一个标量
- `X` 是 (m, n) 矩阵，一行一个样本

**注意：这不叫"多元线性回归 multivariate linear regression"**，Andrew 特别强调正确名称是 **multiple linear regression**（多个输入特征，单个输出）。multivariate regression 在统计学里指多个输出，是另一回事。

## 2. 向量化

```python
f = np.dot(w, x) + b        # 向量化
f = 0
for j in range(n):          # 循环版本
    f += w[j] * x[j]
f += b
```

为什么快：NumPy 底层调用 SIMD/BLAS，把 n 次乘法**并行**做掉，再用树形规约求和；循环版本是严格串行的。n 越大差距越大（n=10⁵ 时能差两个数量级）。

梯度更新也一样：

```python
w = w - alpha * dj_dw        # 一行更新全部 n 个参数
```

矩阵形式的梯度（本仓库用的就是这个）：

```
err   = X @ w + b − y        # (m,)
dj_dw = Xᵀ @ err / m         # (n,)
dj_db = mean(err)            # 标量
```

推导：`∂J/∂wⱼ = (1/m) Σᵢ errᵢ · xᵢⱼ`，对所有 j 打包就是 `Xᵀ err / m`。

## 3. 特征缩放

### 为什么必须做

假设 x₁ = 房屋面积 ∈ [300, 2000]，x₂ = 卧室数 ∈ [0, 5]。

- x₁ 很大 → 它对应的 w₁ 只要变一点点，预测就变很多 → J 对 w₁ 极其敏感
- x₂ 很小 → w₂ 要变很多才影响预测 → J 对 w₂ 很不敏感

等高线是一个**又窄又长的椭圆**。梯度下降的步长在两个方向是同一个 α：沿窄方向要么震荡，沿长方向爬得极慢。**它会来回横跳，走很久才到底。**

缩放后等高线变成近似圆形，梯度直指圆心，几步就到。

### 三种做法

| 方法 | 公式 | 结果范围 |
|---|---|---|
| 除以最大值 | `x/max` | [0, 1] 或 [-1, 1] |
| 均值归一化 | `(x − μ)/(max − min)` | 大约 [-0.5, 0.5]，中心在 0 |
| **Z-score 标准化** | `(x − μ)/σ` | 均值 0，标准差 1 ★最常用 |

经验法则：把每个特征弄到大约 **[-1, 1]** 附近就行，不用精确。
- [0, 3]、[-2, 0.5] → 不用缩放
- [-100, 100]、[-0.0001, 0.0001]、[98.6, 105] → 需要缩放（第三个虽然范围窄，但离 0 很远，也要）

### ★ 最容易犯的错

**μ 和 σ 只能在训练集上算，然后原样用到 cv/test 集上。**

理由：模型是在"缩放后的坐标系"里训练的，新数据必须映射到**同一个**坐标系。在 test 集上重新 fit 一遍等于换了坐标系，学到的 w 就对不上了，而且用到了测试集信息（信息泄漏）。

```python
scaler = ZScoreScaler().fit(X_train)      # 只 fit 训练集
X_train_s = scaler.transform(X_train)
X_cv_s    = scaler.transform(X_cv)        # 复用同一组 mu/sigma
X_test_s  = scaler.transform(X_test)
```

`mlkit/preprocessing.py` 的 `ZScoreScaler` 就是按这个设计的，`fit` 和 `transform` 是分开的。

### 缩放后参数怎么解读

缩放后学到的 w 是"缩放坐标系"下的。要还原到原始单位：`w_orig = w_scaled / σ`，`b_orig = b_scaled − Σ w_scaled·μ/σ`。做完项目要向别人解释系数时会用到。

## 4. 判断收敛

两种办法：
1. **画学习曲线**（J vs 迭代次数），看曲线拉平了没。★推荐，因为还能顺便发现 α 有没有取错。
2. 自动收敛测试：`|ΔJ| < ε`（比如 ε=1e-3）就停。缺点是 ε 难选。

调 α 的实用做法：试 `0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1`（大约每次 ×3），各跑几十轮看 J 曲线，选**能稳定下降的最大那个**。

## 5. 特征工程

用领域知识造新特征。经典例子：已有 `frontage`（临街宽）和 `depth`（进深），造 `area = frontage × depth`。这个新特征往往比原来两个都好用，因为它才是真正决定价格的量。

**这一步的价值经常超过换更复杂的模型。**

## 6. 多项式回归

```
f(x) = w₁x + w₂x² + w₃x³ + b
```

这仍然是**线性回归**——线性指的是"对参数 w 线性"，不是对 x 线性。所以同一套梯度下降代码原样能用，只需要先把 `x, x², x³` 当成三个特征喂进去。

```python
from mlkit.preprocessing import polynomial_features, ZScoreScaler
X_poly = polynomial_features(x, degree=3)     # 列变成 [x, x², x³]
X_poly = ZScoreScaler().fit_transform(X_poly) # 必须缩放！
```

**多项式回归让特征缩放从"建议"变成"必须"**：如果 x ∈ [1, 1000]，那 x³ ∈ [1, 10⁹]，三个特征差 9 个数量级，不缩放梯度下降根本没法跑。

还可以用 `√x`（增长变缓、永不下降，适合"边际收益递减"的关系）。

## 7. 正规方程（Normal Equation）

```
w = (XᵀX)⁻¹Xᵀy
```

- **只对线性回归成立**，逻辑回归/神经网络都没有闭式解
- 不需要迭代，不需要选 α，不需要特征缩放
- 但求逆是 O(n³)，n 上万就慢得不能用
- sklearn 的 `LinearRegression` 底层用的就是这类方法（实际是 SVD/lstsq）

本仓库 `normal_equation()` 用 `np.linalg.lstsq` 而不是显式求逆：当特征完全共线时 `XᵀX` 奇异，显式求逆会报错，lstsq 给出最小范数解。实测和 sklearn 的差在 **3e-15**（浮点极限）。

---

## 8. 代码对照

```python
from mlkit import datasets, preprocessing as pp, linear_regression as lr

X, y = datasets.house_prices()          # 特征范围差两个数量级
scaler = pp.ZScoreScaler().fit(X)
Xs = scaler.transform(X)

w, b, hist = lr.gradient_descent(Xs, y, alpha=0.3, num_iters=1000)
# 未缩放时 alpha 只能到 1e-7；缩放后 0.3 都能稳定下降 —— 这就是缩放的全部意义
```
