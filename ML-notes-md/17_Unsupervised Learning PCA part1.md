# Unsupervised Learning: PCA(Ⅰ)

> 本文將主要介紹 PCA 算法的數學推導過程

上一篇文章提到，PCA 算法認為降維就是一個簡單的 linear function，它的 input x 和 output z 之間是 linear transform，即$z=Wx$，PCA 要做的，就是根據$x$**把 W 給找出來**($z$未知)

#### PCA for 1-D

為了簡化問題，這裡我們假設 z 是 1 維的 vector，也就是把 x 投影到一維空間，此時 w 是一個 row vector

$z_1=w^1\cdot x$，其中$w^1$表示$w$的第一個 row vector，假設$w^1$的長度為 1，即$||w^1||_2=1$，此時$z_1$就是$x$在$w^1$方向上的投影

那我們到底要找什麼樣的$w^1$呢？

假設我們現在已有的寶可夢樣本點分布如下，橫坐標代表寶可夢的攻擊力，縱坐標代表防禦力，我們的任務是把這個二維分布投影到一維空間上

我們希望選這樣一個$w^1$，它使得$x$經過投影之後得到的$z_1$分布越大越好，也就是說，經過這個投影後，不同樣本點之間的區別，應該仍然是可以被看得出來的，即：

- 我們希望找一個 projection 的方向，它可以讓 projection 後的 variance 越大越好

- 我們不希望 projection 使這些 data point 通通擠在一起，導致點與點之間的奇異度消失
- 其中，variance 的計算公式：$Var(z_1)=\frac{1}{N}\sum\limits_{z_1}(z_1-\bar{z_1})^2, ||w^1||_2=1$，$\bar {z_1}$是$z_1$的平均值

下圖給出了所有樣本點在兩個不同的方向上投影之後的 variance 比較情況

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/PCA1.png" width="60%"/></center>

#### PCA for n-D

當然我們不可能只投影到一維空間，我們還可以投影到更高維的空間

對$z=Wx$來說：

- $z_1=w^1\cdot x$，表示$x$在$w^1$方向上的投影
- $z_2=w^2\cdot x$，表示$x$在$w^2$方向上的投影
- ...

$z_1,z_2,...$串起來就得到$z$，而$w^1,w^2,...$分別是$W$的第 1,2,...個 row，需要注意的是，這裡的$w^i$必須相互正交，此時$W$是正交矩陣(orthogonal matrix)，如果不加以約束，則找到的$w^1,w^2,...$實際上是相同的值

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/PCA2.png" width="60%"/></center>

#### Lagrange multiplier

求解 PCA，實際上已經有現成的函數可以調用，此外你也可以把 PCA 描述成 neural network，然後用 gradient descent 的方法來求解，這裡主要介紹用拉格朗日乘數法(Lagrange multiplier)求解 PCA 的數學推導過程

注：$w^i$和$x$均為列向量，下文中類似$w^i\cdot x$表示的是矢量內積，而$(w^i)^T\cdot x$表示的是矩陣相乘

##### calculate $w^1$

目標：maximize $(w^1)^TSw^1 $，條件：$(w^1)^Tw^1=1$

- 首先計算出$\bar{z_1}$：

  $$
  \begin{split}
  &z_1=w^1\cdot x\\
  &\bar{z_1}=\frac{1}{N}\sum z_1=\frac{1}{N}\sum w^1\cdot x=w^1\cdot \frac{1}{N}\sum x=w^1\cdot \bar x
  \end{split}
  $$

- 然後計算 maximize 的對象$Var(z-1)$：

  其中$Cov(x)=\frac{1}{N}\sum(x-\bar x)(x-\bar x)^T$

  $$
  \begin{split}
  Var(z_1)&=\frac{1}{N}\sum\limits_{z_1} (z_1-\bar{z_1})^2\\
  &=\frac{1}{N}\sum\limits_{x} (w^1\cdot x-w^1\cdot \bar x)^2\\
  &=\frac{1}{N}\sum (w^1\cdot (x-\bar x))^2\\
  &=\frac{1}{N}\sum(w^1)^T(x-\bar x)(x-\bar x)^T w^1\\
  &=(w^1)^T\frac{1}{N}\sum(x-\bar x)(x-\bar x)^T w^1\\
  &=(w^1)^T Cov(x)w^1
  \end{split}
  $$

- 當然這裡想要求$Var(z_1)=(w^1)^TCov(x)w^1$的最大值，還要加上$||w^1||_2=(w^1)^Tw^1=1$的約束條件，否則$w^1$可以取無窮大
- 令$S=Cov(x)$，它是：
  - 對稱的(symmetric)
  - 半正定的(positive-semidefine)
  - 所有特徵值(eigenvalues)非負的(non-negative)
- 使用拉格朗日乘數法，利用目標和約束條件構造函數：

  $$
  g(w^1)=(w^1)^TSw^1-\alpha((w^1)^Tw^1-1)
  $$

- 對$w^1$這個 vector 里的每一個 element 做偏微分：

  $$
  \partial g(w^1)/\partial w_1^1=0\\
  \partial g(w^1)/\partial w_2^1=0\\
  \partial g(w^1)/\partial w_3^1=0\\
  ...
  $$

- 整理上述推導式，可以得到：

  其中，$w^1$是 S 的特徵向量(eigenvector)

  $$
  Sw^1=\alpha w^1
  $$

- 注意到滿足$(w^1)^Tw^1=1$的特徵向量$w^1$有很多，我們要找的是可以 maximize $(w^1)^TSw^1$的那一個，於是利用上一個式子：

  $$
  (w^1)^TSw^1=(w^1)^T \alpha w^1=\alpha (w^1)^T w^1=\alpha
  $$

- 此時 maximize $(w^1)^TSw^1$就變成了 maximize $\alpha$，也就是當$S$的特徵值$\alpha$最大時對應的那個特徵向量$w^1$就是我們要找的目標

- 結論：**$w^1$是$S=Cov(x)$這個 matrix 中的特徵向量，對應最大的特徵值$\lambda_1$**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cov.png" width="60%"/></center>

##### calculate $w^2$

在推導$w^2$時，相較於$w^1$，多了一個限制條件：$w^2$必須與$w^1$正交(orthogonal)

目標：maximize $(w^2)^TSw^2$，條件：$(w^2)^Tw^2=1,(w^2)^Tw^1=0$

結論：**$w^2$也是$S=Cov(x)$這個 matrix 中的特徵向量，對應第二大的特徵值$\lambda_2$**

- 同樣是用拉格朗日乘數法求解，先寫一個關於$w^2$的 function，包含要 maximize 的對象，以及兩個約束條件

  $$
  g(w^2)=(w^2)^TSw^2-\alpha((w^2)^Tw^2-1)-\beta((w^2)^Tw^1-0)
  $$

- 對$w^2$的每個 element 做偏微分：

  $$
  \partial g(w^2)/\partial w_1^2=0\\
  \partial g(w^2)/\partial w_2^2=0\\
  \partial g(w^2)/\partial w_3^2=0\\
  ...
  $$

- 整理後得到：

  $$
  Sw^2-\alpha w^2-\beta w^1=0
  $$

- 上式兩側同乘$(w^1)^T$，得到：

  $$
  (w^1)^TSw^2-\alpha (w^1)^Tw^2-\beta (w^1)^Tw^1=0
  $$

- 其中$\alpha (w^1)^Tw^2=0,\beta (w^1)^Tw^1=\beta$，

  而由於$(w^1)^TSw^2$是 vector×matrix×vector=scalar，因此在外面套一個 transpose 不會改變其值，因此該部分可以轉化為：

  注：S 是 symmetric 的，因此$S^T=S$

  $$
  \begin{split}
  (w^1)^TSw^2&=((w^1)^TSw^2)^T\\
  &=(w^2)^TS^Tw^1\\
  &=(w^2)^TSw^1
  \end{split}
  $$

  我們已經知道$w^1$滿足$Sw^1=\lambda_1 w^1$，代入上式：

  $$
  \begin{split}
  (w^1)^TSw^2&=(w^2)^TSw^1\\
  &=\lambda_1(w^2)^Tw^1\\
  &=0
  \end{split}
  $$

- 因此有$(w^1)^TSw^2=0$，$\alpha (w^1)^Tw^2=0$，$\beta (w^1)^Tw^1=\beta$，又根據

  $$
  (w^1)^TSw^2-\alpha (w^1)^Tw^2-\beta (w^1)^Tw^1=0
  $$

  可以推得$\beta=0$

- 此時$Sw^2-\alpha w^2-\beta w^1=0$就轉變成了$Sw^2-\alpha w^2=0$，即

  $$
  Sw^2=\alpha w^2
  $$

- 由於$S$是 symmetric 的，因此在不與$w_1$衝突的情況下，這裡$\alpha$選取第二大的特徵值$\lambda_2$時，可以使$(w^2)^TSw^2$最大

- 結論：**$w^2$也是$S=Cov(x)$這個 matrix 中的特徵向量，對應第二大的特徵值$\lambda_2$**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cov2.png" width="60%"/></center>

#### PCA-decorrelation

$z=W\cdot x$

神奇之處在於$Cov(z)=D$，即 z 的 covariance 是一個 diagonal matrix，推導過程如下圖所示

PCA 可以讓不同 dimension 之間的 covariance 變為 0，即不同 new feature 之間是沒有 correlation 的，這樣做的好處是，**減少 feature 之間的聯繫從而減少 model 所需的參數量**

如果你把原來的 input data 通過 PCA 之後再給其他 model 使用，那這些 model 就可以使用簡單的形式，而無需考慮不同 dimension 之間類似$x_1\cdot x_2,x_3\cdot x_5^3,...$這些交叉項，此時 model 得到簡化，參數量大大降低，相同的 data 量可以得到更好的訓練結果，從而可以避免 overfitting 的發生

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cov3.png" width="60%"/></center>

本文主要介紹的是 PCA 的數學推導，如果你理解起來有點困難，那下一篇文章將會從另一個角度解釋 PCA 算法的原理~
