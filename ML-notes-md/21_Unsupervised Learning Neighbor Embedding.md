# Unsupervised Learning: Neighbor Embedding

> 本文介紹了非線性降維的一些算法，包括局部線性嵌入 LLE、拉普拉斯特徵映射和 t 分布隨機鄰居嵌入 t-SNE，其中 t-SNE 特別適用於可視化的應用場景

PCA 和 Word Embedding 介紹了線性降維的思想，而 Neighbor Embedding 要介紹的是非線性的降維

#### Manifold Learning

樣本點的分布可能是在高維空間里的一個流行(Manifold)，也就是說，樣本點其實是分布在低維空間裡面，只是被扭曲地塞到了一個高維空間里

地球的表面就是一個流行(Manifold)，它是一個二維的平面，但是被塞到了一個三維空間里

在 Manifold 中，只有距離很近的點歐氏距離(Euclidean Distance)才會成立，而在下圖的 S 型曲面中，歐氏距離是無法判斷兩個樣本點的相似程度的

而 Manifold Learning 要做的就是把這個 S 型曲面降維展開，把塞在高維空間里的低維空間攤平，此時使用歐氏距離就可以描述樣本點之間的相似程度

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/manifold.png" width="60%" /></center>

#### Locally Linear Embedding

局部線性嵌入，locally linear embedding，簡稱**LLE**

假設在原來的空間中，樣本點的分布如下所示，我們關注$x^i$和它的鄰居$x^j$，用$w_{ij}$來描述$x_i$和$x_j$的關係

假設每一個樣本點$x^i$都是可以用它的 neighbor 做 linear combination 組合而成，那$w_{ij}$就是拿$x^j$去組合$x^i$時的權重 weight，因此找點與點的關係$w_{ij}$這個問題就轉換成，找一組使得所有樣本點與周圍點線性組合的差距能夠最小的參數$w_{ij}$：

$$
\sum\limits_i||x^i-\sum\limits_j w_{ij}x^j ||_2
$$

接下來就要做 Dimension Reduction，把$x^i$和$x^j$降維到$z^i$和$z^j$，並且保持降維前後兩個點之間的關係$w_{ij}$是不變的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lle.png" width="60%" /></center>

LLE 的具體做法如下：

- 在原先的高維空間中找到$x^i$和$x^j$之間的關係$w_{ij}$以後就把它固定住

- 使$x^i$和$x^j$降維到新的低維空間上的$z^i$和$z^j$

- $z^i$和$z^j$需要 minimize 下面的式子：

  $$
  \sum\limits_i||z^i-\sum\limits_j w_{ij}z^j ||_2
  $$

- 即在原本的空間里，$x^i$可以由周圍點通過參數$w_{ij}$進行線性組合得到，則要求在降維後的空間里，$z^i$也可以用同樣的線性組合得到

實際上，LLE 並沒有給出明確的降維函數，它沒有明確地告訴我們怎麼從$x^i$降維到$z^i$，只是給出了降維前後的約束條件

在實際應用 LLE 的時候，對$x^i$來說，需要選擇合適的鄰居點數目 K 才會得到好的結果

下圖給出了原始 paper 中的實驗結果，K 太小或太大得到的結果都不太好，注意到在原先的空間里，只有距離很近的點之間的關係需要被保持住，如果 K 選的很大，就會選中一些由於空間扭曲才導致距離接近的點，而這些點的關係我們並不希望在降維後還能被保留

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lle2.png" width="60%" /></center>

#### Laplacian Eigenmaps

##### Introduction

另一種方法叫拉普拉斯特徵映射，Laplacian Eigenmaps

之前在 semi-supervised learning 有提到 smoothness assumption，即我們僅知道兩點之間的歐氏距離是不夠的，還需要觀察兩個點在 high density 區域下的距離

如果兩個點在 high density 的區域里比較近，那才算是真正的接近

我們依據某些規則把樣本點建立 graph，那麼 smoothness 的距離就可以使用 graph 中連接兩個點路徑上的 edges 數來近似

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/le.png" width="60%" /></center>

##### Review for Smoothness Assumption

簡單回顧一下在 semi-supervised 里的說法：如果兩個點$x^1$和$x^2$在高密度區域上是相近的，那它們的 label $y^1$和$y^2$很有可能是一樣的

$$
L=\sum\limits_{x^r} C(y^r,\hat y^r) + \lambda S\\
S=\frac{1}{2}\sum\limits_{i,j} w_{i,j}(y^i-y^j)^2=y^TLy
$$

其中$C(y^r,\hat y^r)$表示 labeled data 項，$\lambda S$表示 unlabeled data 項，它就像是一個 regularization term，用於判斷我們當前得到的 label 是否是 smooth 的

其中如果點$x^i$與$x^j$是相連的，則$w_{i,j}$等於相似度，否則為 0，$S$的表達式希望在$x^i$與$x^j$很接近的情況下，相似度$w_{i,j}$很大，而 label 差距$|y^i-y^j|$越小越好，同時也是對 label 平滑度的一個衡量

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/le2.png" width="60%" /></center>

##### Application in Unsupervised Task

降維的基本原則：如果$x^i$和$x^j$在 high density 區域上是相近的，即相似度$w_{i,j}$很大，則降維後的$z^i$和$z^j$也需要很接近，總體來說就是讓下面的式子盡可能小

$$
S=\frac{1}{2}\sum\limits_{i,j} w_{i,j}(y^i-y^j)^2
$$

注意，與 LLE 不同的是，這裡的$w_{i,j}$表示$x^i$與$x^j$這兩點的相似度，上式也可以寫成$S=\sum\limits_{i,j} w_{i,j} ||z^i-z^j||_2$

但光有上面這個式子是不夠的，假如令所有的 z 相等，比如令$z^i=z^j=0$，那上式就會直接停止更新

在 semi-supervised 中，如果所有 label $z^i$都設成一樣，會使得 supervised 部分的$\sum\limits_{x^r} C(y^r,\hat y^r)$變得很大，因此 lost 就會很大，但在這裡少了 supervised 的約束，因此我們需要給$z$一些額外的約束：

- 假設降維後$z$所處的空間為$M$維，則$\{z^1,z^2,...,z^N\}=R^M$，我們希望降維後的$z$佔據整個$M$維的空間，而不希望它活在一個比$M$更低維的空間里
- 最終解出來的$z$其實就是 Graph Laplacian $L$比較小的特徵值所對應的特徵向量

這也是 Laplacian Eigenmaps 名稱的由來，我們找的$z$就是 Laplacian matrix 的特徵向量

如果通過拉普拉斯特徵映射找到$z$之後再對其利用 K-means 做聚類，就叫做譜聚類(spectral clustering)

注：有關拉普拉斯圖矩陣的相關內容可參考之前的半監督學習筆記：[15_Semi-supervised Learning](https://sakura-gh.github.io/ML-notes/ML-notes-html/15_Semi-supervised-Learning.html)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/le3.png" width="60%" /></center>

參考文獻：_Belkin, M., Niyogi, P. Laplacian eigenmaps and spectral techniques for embedding and clustering. Advances in neural information processing systems . 2002_

#### t-SNE

t-SNE，全稱為 T-distributed Stochastic Neighbor Embedding，t 分布隨機鄰居嵌入

##### Shortage in LLE

前面的方法**只假設了相鄰的點要接近，卻沒有假設不相近的點要分開**

所以在 MNIST 使用 LLE 會遇到下圖的情形，它確實會把同一個 class 的點都聚集在一起，卻沒有辦法避免不同 class 的點重疊在一個區域，這就會導致依舊無法區分不同 class 的現象

COIL-20 數據集包含了同一張圖片進行旋轉之後的不同形態，對其使用 LLE 降維後得到的結果是，同一個圓圈代表同張圖像旋轉的不同姿態，但許多圓圈之間存在重疊

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/tsne.png" width="60%" /></center>

##### How t-SNE works

做 t-SNE 同樣要降維，在原來$x$的分布空間上，我們需要計算所有$x^i$與$x^j$之間的相似度$S(x^i,x^j)$

然後需要將其做歸一化：$P(x^j|x^i)=\frac{S(x^i,x^j)}{\sum_{k\ne i}S(x^i,x^k)}$，即$x^j$與$x^i$的相似度佔所有與$x^i$相關的相似度的比例

將$x$降維到$z$，同樣可以計算相似度$S'(z^i,z^j)$，並做歸一化：$Q(z^j|z^i)=\frac{S'(z^i,z^j)}{\sum_{k\ne i}S'(z^i,z^k)}$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/tsne2.png" width="60%" /></center>

注意，這裡的歸一化是有必要的，因為我們無法判斷在$x$和$z$所在的空間里，$S(x^i,x^j)$與$S'(z^i,z^j)$的範圍是否是一致的，需要將其映射到一個統一的概率區間

我們希望找到的投影空間$z$，可以讓$P(x^j|x^i)$和$Q(z^j|z^i)$的分布越接近越好

用於衡量兩個分布之間相似度的方法就是**KL 散度(KL divergence)**，我們的目標就是讓$L$越小越好：

$$
L=\sum\limits_i KL(P(*|x^i)||Q(*|z^i))\\
=\sum\limits_i \sum\limits_jP(x^j|x^i)log \frac{P(x^j|x^i)}{Q(z^j|z^i)}
$$

##### KL Divergence

這裡簡單補充一下 KL 散度的基本知識

KL 散度，最早是從信息論里演化而來的，所以在介紹 KL 散度之前，我們要先介紹一下信息熵，信息熵的定義如下：

$$
H=-\sum\limits_{i=1}^N p(x_i)\cdot log\ p(x_i)
$$

其中$p(x_i)$表示事件$x_i$發生的概率，信息熵其實反映的就是要表示一個概率分布所需要的平均信息量

在信息熵的基礎上，我們定義 KL 散度為：

$$
D_{KL}(p||q)=\sum\limits_{i=1}^N p(x_i)\cdot (log\ p(x_i)-log\ q(x_i))\\
=\sum\limits_{i=1}^N p(x_i)\cdot log\frac{p(x_i)}{q(x_i)}
$$

$D_{KL}(p||q)$表示的就是概率$q$與概率$p$之間的差異，很顯然，KL 散度越小，說明概率$q$與概率$p$之間越接近，那麼預測的概率分布與真實的概率分布也就越接近

##### How to use

t-SNE 會計算所有樣本點之間的相似度，運算量會比較大，當數據量大的時候跑起來效率會比較低

常見的做法是對原先的空間用類似 PCA 的方法先做一次降維，然後用 t-SNE 對這個簡單降維空間再做一次更深層次的降維，以期減少運算量

值得注意的是，t-SNE 的式子無法對新的樣本點進行處理，一旦出現新的$x^i$，就需要重新跑一遍該算法，所以**t-SNE 通常不是用來訓練模型的，它更適合用於做基於固定數據的可視化**

t-SNE 常用於將固定的高維數據可視化到二維平面上

##### Similarity Measure

如果根據歐氏距離計算降維前的相似度，往往採用**RBF function** $S(x^i,x^j)=e^{-||x^i-x^j||_2}$，這個表達式的好處是，只要兩個樣本點的歐氏距離稍微大一些，相似度就會下降得很快

還有一種叫做 SNE 的方法，它在降維後的新空間採用與上述相同的相似度算法$S'(z^i,z^j)=e^{-||z^i-z^j||_2}$

對 t-SNE 來說，它在降維後的新空間所採取的相似度算法是與之前不同的，它選取了**t-distribution**中的一種，即$S'(z^i,z^j)=\frac{1}{1+||z^i-z^j||_2}$

以下圖為例，假設橫軸代表了在原先$x$空間上的歐氏距離或者做降維之後在$z$空間上的歐氏距離，紅線代表 RBF function，是降維前的分布；藍線代表了 t-distribution，是降維後的分布

你會發現，降維前後相似度從 RBF function 到 t-distribution：

- 如果原先兩個點距離($\Delta x$)比較近，則降維轉換之後，它們的相似度($\Delta y$)依舊是比較接近的
- 如果原先兩個點距離($\Delta x$)比較遠，則降維轉換之後，它們的相似度($\Delta y$)會被拉得更遠

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/tsne3.png" width="60%" /></center>

也就是說 t-SNE 可以聚集相似的樣本點，同時還會放大不同類別之間的距離，從而使得不同類別之間的分界線非常明顯，特別適用於可視化，下圖則是對 MNIST 和 COIL-20 先做 PCA 降維，再做 t-SNE 降維可視化的結果：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/tsne4.png" width="60%" /></center>

#### Conclusion

小結一下，本文主要介紹了三種非線性降維的算法：

- LLE(Locally Linear Embedding)，局部線性嵌入算法，主要思想是降維前後，每個點與周圍鄰居的線性組合關係不變，$x^i=\sum\limits_j w_{ij}x^j$、$z^i=\sum\limits_j w_{ij}z^j$
- Laplacian Eigenmaps，拉普拉斯特徵映射，主要思想是在 high density 的區域，如果$x^i$、$x^j$這兩個點相似度$w_{i,j}$高，則投影後的距離$||z^i-z^j||_2$要小
- t-SNE(t-distribution Stochastic Neighbor Embedding)，t 分布隨機鄰居嵌入，主要思想是，通過降維前後計算相似度由 RBF function 轉換為 t-distribution，在聚集相似點的同時，拉開不相似點的距離，比較適合用在數據固定的可視化領域
