# Unsupervised Learning: PCA(Ⅱ)

> 本文主要從組件和 SVD 分解的角度介紹 PCA，並描述了 PCA 的神經網絡實現方式，通過引入寶可夢、手寫數字分解、人臉圖像分解的例子，介紹了 NMF 算法的基本思想，此外，還提供了一些 PCA 相關的降維算法和論文

#### Reconstruction Component

假設我們現在考慮的是手寫數字識別，這些數字是由一些類似於筆畫的 basic component 組成的，本質上就是一個 vector，記做$u_1,u_2,u_3,...$，以 MNIST 為例，不同的筆畫都是一個 28×28 的 vector，把某幾個 vector 加起來，就組成了一個 28×28 的 digit

寫成表達式就是：$x≈c_1u^1+c_2u^2+...+c_ku^k+\bar x$

其中$x$代表某張 digit image 中的 pixel，它等於 k 個 component 的加權和$\sum c_iu^i$加上所有 image 的平均值$\bar x$

比如 7 就是$x=u^1+u^3+u^5$，我們可以用$\left [\begin{matrix}c_1\ c_2\ c_3...c_k \end{matrix} \right]^T$來表示一張 digit image，如果 component 的數目 k 遠比 pixel 的數目要小，那這個描述就是比較有效的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bc.png" width="60%"/></center>

實際上目前我們並不知道$u^1$~$u^k$具體的值，因此我們要找這樣 k 個 vector，使得$x-\bar x$與$\hat x$越接近越好：

$$
x-\bar x≈c_1u^1+c_2u^2+...+c_ku^k=\hat x
$$

而用未知 component 來描述的這部分內容，叫做 Reconstruction error，即$||(x-\bar x)-\hat x||$

接下來我們就要去找 k 個 vector $u^i$去 minimize 這個 error：

$$
L=\min\limits_{u^1,...,u^k}\sum||(x-\bar x)-(\sum\limits_{i=1}^k c_i u^i) ||_2
$$

回顧 PCA，$z=W\cdot x$，實際上我們通過 PCA 最終解得的$\{w^1,w^2,...,w^k\}$就是使 reconstruction error 最小化的$\{u^1,u^2,...,u^k\}$，簡單證明如下：

- 我們將所有的$x^i-\bar x≈c_1^i u^1+c_2^i u^2+...$都用下圖中的矩陣相乘來表示，我們的目標是使等號兩側矩陣之間的差距越小越好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/re.png" width="60%"/></center>

- 可以使用 SVD 將每個 matrix $X_{m×n}$都拆成 matrix $U_{m×k}$、$\Sigma_{k×k}$、$V_{k×n}$的乘積，其中 k 為 component 的數目
- 值得注意的是，使用 SVD 拆解後的三個矩陣相乘，是跟等號左邊的矩陣$X$最接近的，此時$U$就對應著$u^i$那部分的矩陣，$\Sigma\cdot V$就對應著$c_k^i$那部分的矩陣
- 根據 SVD 的結論，組成矩陣$U$的 k 個列向量(標準正交向量, orthonormal vector)就是$XX^T$最大的 k 個特徵值(eignvalue)所對應的特徵向量(eigenvector)，而$XX^T$實際上就是$x$的 covariance matrix，因此$U$就是 PCA 的 k 個解
- 因此我們可以發現，通過 PCA 找出來的 Dimension Reduction 的 transform，實際上就是把$X$拆解成能夠最小化 Reconstruction error 的 component 的過程，通過 PCA 所得到的$w^i$就是 component $u^i$，而 Dimension Reduction 的結果就是參數$c_i$
- 簡單來說就是，用 PCA 對$x$進行降維的過程中，我們要找的投影方式$w^i$就相當於恰當的組件$u^i$，投影結果$z^i$就相當於這些組件各自所佔的比例$c_i$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svd.png" width="60%"/></center>

- 下面的式子簡單演示了將一個樣本點$x$劃分為 k 個組件的過程，其中$\left [\begin{matrix}c_1 \ c_2\ ... c_k \end{matrix} \right ]^T$是每個組件的比例；把$x$劃分為 k 個組件即從 n 維投影到 k 維空間，$\left [\begin{matrix}c_1 \ c_2\ ... c_k \end{matrix} \right ]^T$也是投影結果

  注：$x$和$u_i$均為 n 維列向量

  $$
  \begin{split}
  &x=
  \left [
  \begin{matrix}
  u_1\ u_2\ ...\ u_k
  \end{matrix}
  \right ]\cdot
  \left [
  \begin{matrix}
  c_1\\
  c_2\\
  ...\\
  c_k
  \end{matrix}
  \right ]\\ \\

  &\left [
  \begin{matrix}
  x_1\\
  x_2\\
  ...\\
  x_n
  \end{matrix}
  \right ]=\left [
  \begin{matrix}
  u_1^1\ u_2^1\ ... u_k^1 \\
  u_1^2\ u_2^2\ ... u_k^2 \\
  ...\\
  u_1^n\ u_2^n\ ... u_k^n
  \end{matrix}
  \right ]\cdot
  \left [
  \begin{matrix}
  c_1\\
  c_2\\
  ...\\
  c_k
  \end{matrix}
  \right ]\\
  \end{split}
  $$

#### NN for PCA

現在我們已經知道，用 PCA 找出來的$\{w^1,w^2,...,w^k\}$就是 k 個 component $\{u^1,u^2,...,u^k\}$

而$\hat x=\sum\limits_{k=1}^K c_k w^k$，我們要使$\hat x$與$x-\bar x$之間的差距越小越好，我們已經根據 SVD 找到了$w^k$的值，而對每個不同的樣本點，都會有一組不同的$c_k$值

在 PCA 中我們已經證得，$\{w^1,w^2,...,w^k\}$這 k 個 vector 是標準正交化的(orthonormal)，因此：

$$
c_k=(x-\bar x)\cdot w^k
$$

這個時候我們就可以使用神經網絡來表示整個過程，假設$x$是 3 維向量，要投影到 k=2 維的 component 上：

- 對$x-\bar x$與$w^k$做 inner product 的過程類似於 neural network，$x-\bar x$在 3 維空間上的坐標就相當於是 neuron 的 input，而$w^1_1$，$w^1_2$，$w^1_3$則是 neuron 的 weight，表示在$w^1$這個維度上投影的參數，而$c_1$則是這個 neuron 的 output，表示在$w^1$這個維度上投影的坐標值；對$w^2$也同理

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pca-nn.png" width="60%"/></center>

- 得到$c_1$之後，再讓它乘上$w^1$，得到$\hat x$的一部分

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pca-nn2.png" width="60%"/></center>

- 對$c_2$進行同樣的操作，乘上$w^2$，貢獻$\hat x$的剩餘部分，此時我們已經完整計算出$\hat x$三個分量的值

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pca-nn3.png" width="60%"/></center>

- 此時，PCA 就被表示成了只含一層 hidden layer 的神經網絡，且這個 hidden layer 是線性的激活函數，訓練目標是讓這個 NN 的 input $x-\bar x$與 output $\hat x$越接近越好，這件事就叫做**Autoencoder**

- 注意，通過 PCA 求解出的$w^i$與直接對上述的神經網絡做梯度下降所解得的$w^i$是會不一樣的，因為 PCA 解出的$w^i$是相互垂直的(orgonormal)，而用 NN 的方式得到的解無法保證$w^i$相互垂直，NN 無法做到 Reconstruction error 比 PCA 小，因此：
  - 在 linear 的情況下，直接用 PCA 找$W$遠比用神經網絡的方式更快速方便
  - 用 NN 的好處是，它可以使用不止一層 hidden layer，它可以做**deep** autoencoder

#### Weakness of PCA

PCA 有很明顯的弱點：

- 它是**unsupervised**的，如果我們要將下圖綠色的點投影到一維空間上，PCA 給出的從左上到右下的劃分很有可能使原本屬於藍色和橙色的兩個 class 的點被 merge 在一起

  而 LDA 則是考慮了 labeled data 之後進行降維的一種方式，但屬於 supervised

- 它是**linear**的，對於下圖中的彩色曲面，我們期望把它平鋪拉直進行降維，但這是一個 non-linear 的投影轉換，PCA 無法做到這件事情，PCA 只能做到把這個曲面打扁壓在平面上，類似下圖，而無法把它拉開

  對類似曲面空間的降維投影，需要用到 non-linear transformation

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pca-weak.png" width="60%"/></center>

#### PCA for Pokemon

這裡舉一個實際應用的例子，用 PCA 來分析寶可夢的數據

假設總共有 800 只寶可夢，每只都是一個六維度的樣本點，即 vector={HP, Atk, Def, Sp Atk, Sp Def, Speed}，接下來的問題是，我們要投影到多少維的空間上？

如果做可視化分析的話，投影到二維或三維平面可以方便人眼觀察

實際上，寶可夢的$cov(x)$是 6 維，最多可以投影到 6 維空間，我們可以先找出 6 個特徵向量和對應的特徵值$\lambda_i$，其中$\lambda_i$表示第 i 個投影維度的 variance 有多大(即在第 i 個維度的投影上點的集中程度有多大)，然後我們就可以計算出每個$\lambda_i$的比例，ratio=$\frac{\lambda_i}{\sum\limits_{i=1}^6 \lambda_i}$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pca-poke.png" width="60%"/></center>

從上圖的 ratio 可以看出$\lambda_5$、$\lambda_6$所佔比例不高，即第 5 和第 6 個 principle component(可以理解為維度)所發揮的作用是比較小的，用這兩個 dimension 做投影所得到的 variance 很小，投影在這兩個方向上的點比較集中，意味著這兩個維度表示的是寶可夢的共性，無法對區分寶可夢的特性做出太大的貢獻，所以我們只需要利用前 4 個 principle component 即可

注意到新的維度本質上就是舊的維度的加權矢量和，下圖給出了前 4 個維度的加權情況，從 PC1 到 PC4 這 4 個 principle component 都是 6 維度加權的 vector，它們都可以被認為是某種組件，大多數的寶可夢都可以由這 4 種組件拼接而成，也就是用這 4 個 6 維的 vector 做 linear combination 的結果

我們來仔細分析一下這些組件：

- 對第一個 vector PC1 來說，每個值都是正的，因此這個組件在某種程度上代表了寶可夢的強度

- 對第二個 vector PC2 來說，防禦力 Def 很大而速度 Speed 很小，這個組件可以增加寶可夢的防禦力但同時會犧牲一部分的速度

- 如果將寶可夢僅僅投影到 PC1 和 PC2 這兩個維度上，則降維後的二維可視化圖像如下圖所示：

  從該圖中也可以得到一些信息：

  - 在 PC2 維度上特別大的那個樣本點剛好對應著普普(海龜)，確實是防禦力且速度慢的寶可夢
  - 在 PC1 維度上特別大的那三個樣本點則對應著蓋歐卡、超夢等綜合實力很強的寶可夢

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pca-poke2.png" width="60%"/></center>

- 對第三個 vector PC3 來說，sp Def 很大而 HP 和 Atk 很小，這個組件是用生命力和攻擊力來換取特殊防禦力

- 對第四個 vector PC4 來說，HP 很大而 Atk 和 Def 很小，這個組件是用攻擊力和防禦力來換取生命力

- 同樣將寶可夢只投影到 PC3 和 PC4 這兩個維度上，則降維後得到的可視化圖像如下圖所示：

  該圖同樣可以告訴我們一些信息：

  - 在 PC3 維度上特別大的樣本點依舊是普普，第二名是冰柱機器人，它們的特殊防禦力都比較高
  - 在 PC4 維度上特別大的樣本點則是吉利蛋和幸福蛋，它們的生命力比較強

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pca-poke3.png" width="60%"/></center>

#### PCA for MNIST

再次回到手寫數字識別的問題上來，這個時候我們就可以熟練地把一張數字圖像用多個組件(維度)表示出來了：

$$
digit\ image=a_1 w^1+a_2 w^2+...
$$

這裡的$w^i$就表示降維後的其中一個維度，同時也是一個組件，它是由原先 28×28 維進行加權求和的結果，因此$w^i$也是一張 28×28 的圖像，下圖列出了通過 PCA 得到的前 30 個組件的形狀：

注：PCA 就是求$Cov(x)=\frac{1}{N}\sum (x-\bar x)(x-\bar x)^T$的前 30 個最大的特徵值對應的特徵向量

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pca-mnist.png" width="60%"/></center>

#### PCA for Face

同理，通過 PCA 找出人臉的前 30 個組件(維度)，如下圖所示：

用這些臉的組件做線性組合就可以得到所有的臉

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pca-face.png" width="60%"/></center>

#### What happens to PCA

在對 MNIST 和 Face 的 PCA 結果展示的時候，你可能會注意到我們找到的組件好像並不算是組件，比如 MNIST 找到的幾乎是完整的數字雛形，而 Face 找到的也幾乎是完整的人臉雛形，但我們預期的組件不應該是類似於橫折撇捺，眼睛鼻子眉毛這些嗎？

如果你仔細思考了 PCA 的特性，就會發現得到這個結果是可能的

$$
digit\ image=a_1 w^1+a_2 w^2+...
$$

注意到 linear combination 的 weight $a_i$可以是正的也可以是負的，因此我們可以通過把組件進行相加或相減來獲得目標圖像，這會導致你找出來的 component 不是基礎的組件，但是通過這些組件的加加減減肯定可以獲得基礎的組件元素

#### NMF

##### Introduction

如果你要一開始就得到類似筆畫這樣的基礎組件，就要使用 NMF(non-negative matrix factorization)，非負矩陣分解的方法

PCA 可以看成對原始矩陣$X$做 SVD 進行矩陣分解，但並不保證分解後矩陣的正負，實際上當進行圖像處理時，如果部分組件的 matrix 包含一些負值的話，如何處理負的像素值也會成為一個問題(可以做歸一化處理，但比較麻煩)

而 NMF 的基本精神是，強迫使所有組件和它的加權值都必須是正的，也就是說**所有圖像都必須由組件疊加得到**：

- Forcing $a_1$, $a_2$...... be non-negative
  - additive combination
- Forcing $w_1$, $w_2$...... be non-negative
  - More like 「parts of digits」

注：關於 NMF 的具體算法內容可參考 paper(公眾號回復「NMF」獲取 pdf)：

_Daniel D. Lee and H. Sebastian Seung. "Algorithms for non-negative matrix factorization."Advances in neural information processing systems. 2001._

##### NMF for MNIST

在 MNIST 數據集上，通過 NMF 找到的前 30 個組件如下圖所示，可以發現這些組件都是由基礎的筆畫構成：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/nmf-mnist.png" width="60%"/></center>

##### NMF for Face

在 Face 數據集上，通過 NMF 找到的前 30 個組價如下圖所示，相比於 PCA 這裡更像是臉的一部分

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/nmf-face.png" width="60%"/></center>

#### More Related Approaches

降維的方法有很多，這裡再列舉一些與 PCA 有關的方法：

- Multidimensional Scaling (**MDS**) [Alpaydin, Chapter 6.7]

  MDS 不需要把每個 data 都表示成 feature vector，只需要知道特徵向量之間的 distance，就可以做降維，PCA 保留了原來在高維空間中的距離，在某種情況下 MDS 就是特殊的 PCA

- **Probabilistic PCA** [Bishop, Chapter 12.2]

  PCA 概率版本

- **Kernel PCA** [Bishop, Chapter 12.3]

  PCA 非線性版本

- Canonical Correlation Analysis (**CCA**) [Alpaydin, Chapter 6.9]

  CCA 常用於兩種不同的 data source 的情況，比如同時對聲音信號和唇形的圖像進行降維

- Independent Component Analysis (**ICA**)

  ICA 常用於 source separation，PCA 找的是正交的組件，而 ICA 則只需要找「獨立」的組件即可

- Linear Discriminant Analysis (**LDA**) [Alpaydin, Chapter 6.8]

  LDA 是 supervised 的方式
