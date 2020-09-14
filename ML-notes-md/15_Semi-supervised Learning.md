# Semi-supervised Learning

> 半監督學習(semi-supervised learning)
> 1、introduction
> 2、Semi-supervised Learning for Generative Model
> 3、Low-density Separation Assumption：非黑即白
> 4、Smoothness Assumption：近朱者赤，近墨者黑
> 5、Better Representation：去蕪存菁，化繁為簡

#### Introduction

Supervised Learning：$(x^r,\hat y^r)$$_{r=1}^R$

- training data 中，**每一組**data 都有 input $x^r$和對應的 output $y^r$

Semi-supervised Learning：$\{(x^r,\hat y^r)\}_{r=1}^R$} + $\{x^u\}_{u=R}^{R+U}$

- training data 中，部分 data 沒有標籤，只有 input $x^u$

- 通常遇到的場景是，無標籤的數據量遠大於有標籤的數據量，即**U>>R**

- semi-supervised learning 分為以下兩種情況：

  - Transductive Learning：unlabeled data is the testing data

    即，把 testing data 當做無標籤的 training data 使用，適用於事先已經知道 testing data 的情況(一些比賽的時候)

    值得注意的是，這種方法使用的僅僅是 testing data 的**feature**，而不是 label，因此不會出現「直接對 testing data 做訓練而產生 cheating 的效果」

  - Inductive Learning：unlabeled data is not the testing data

    即，不把 testing data 的 feature 拿去給機器訓練，適用於事先並不知道 testing data 的情況(更普遍的情況)

- 為什麼要做 semi-supervised learning？

  實際上我們從來不缺 data，只是缺有 label 的 data，就像你可以拍很多照片，但它們一開始都是沒有標籤的

#### Why semi-supervised learning help？

為什麼 semi-supervised learning 會有效呢？

_The distribution of the unlabeled data tell us something._

unlabeled data 雖然只有 input，但它的**分布**，卻可以告訴我們一些事情

以下圖為例，在只有 labeled data 的情況下，紅線是二元分類的分界線

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/semi-help1.png" width="45%;"/></center>

但當我們加入 unlabeled data 的時候，由於**特徵分布**發生了變化，分界線也隨之改變

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/semi-help2.png" width="50%;"/></center>

semi-supervised learning 的使用往往伴隨著假設，而該假設的合理與否，決定了結果的好壞程度；比如上圖中的 unlabeled data，它顯然是一隻狗，而特徵分布卻與貓被劃分在了一起，很可能是由於這兩張圖片的背景都是綠色導致的，因此假設是否合理顯得至關重要

#### Semi-supervised Learning for Generative Model

##### Supervised Generative Model

事實上，在監督學習中，我們已經討論過概率生成模型了，假設 class1 和 class2 的分布分別為$mean_1=u^1,covariance_1=\Sigma$、$mean_2=u^2,covariance_2=\Sigma$的高斯分布，計算出 Prior Probability 後，再根據貝葉斯公式可以推得新生成的 x 所屬的類別

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/super-gm.png" width="60%;"/></center>

##### Semi-supervised Generative Model

如果在原先的數據下多了 unlabeled data(下圖中綠色的點)，它就會影響最終的決定，你會發現原先的$u,\Sigma$顯然是不合理的，新的$u,\Sigma$需要使得樣本點的分布更接近下圖虛線圓所標出的範圍，除此之外，右側的 Prior Probability 會給人一種比左側大的感覺(右側樣本點"變多"了)

此時，unlabeled data 對$P(C_1),P(C_2),u^1,u^2,\Sigma$都產生了一定程度的影響，劃分兩個 class 的 decision boundary 也會隨之發生變化

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/super-semi.png" width="60%;"/></center>

講完了直觀上的解釋，接下來進行具體推導(假設做二元分類)：

- 先隨機初始化一組參數：$\theta=\{P(C_1),P(C_2),u^1,u^2,\Sigma\}$

- step1：利用初始 model 計算每一筆 unlabeled data $x^u$屬於 class 1 的概率$P_{\theta}(C_1|x^u)$

- step2：update model

  如果不考慮 unlabeled data，則先驗概率顯然為屬於 class1 的樣本點數$N_1$/總的樣本點數$N$，即$P(C_1)=\frac{N_1}{N}$

  而考慮 unlabeled data 時，分子還要加上所有 unlabeled data 屬於 class 1 的概率和，此時它們被看作小數，可以理解為按照概率一部分屬於$C_1$，一部分屬於$C_2$

  $$
  P(C_1)=\frac{N_1+\sum_{x^u}P(C_1|x^u)}{N}
  $$

  同理，對於均值，原先的 mean $u_1=\frac{1}{N_1}\sum\limits_{x^r\in C_1} x^r$加上根據概率對$x^u$求和再歸一化的結果即可

  $$
  u_1=\frac{1}{N_1}\sum\limits_{x^r\in C_1} x^r+\frac{1}{\sum_{x^u}P(C_1|x^u)}\sum\limits_{x^u}P(C_1|x^u)x^u
  $$

  剩餘的參數同理，接下來就有了一組新的參數$\theta'$，於是回到 step1->step2->step1 循環

- 理論上該方法保證是可以收斂的，而一開始給$\theta$的初始值會影響收斂的結果，類似 gradient descent

- 上述的 step1 就是 EM algorithm 里的 E，step2 則是 M

以上的推導基於的基本思想是，把 unlabeled data $x^u$看成是可以劃分的，一部分屬於$C_1$，一部分屬於$C_2$，此時它的概率$P_{\theta}(x^u)=P_{\theta}(x^u|C_1)P(C_1)+P_{\theta}(x^u|C_2)P(C_2)$，也就是$C_1$的先驗概率乘上$C_1$這個 class 產生$x^u$的概率+$C_2$的先驗概率乘上$C_2$這個 class 產生$x^u$的概率

實際上我們在利用極大似然函數更新參數的時候，就利用了該拆分的結果：

$$
logL(\theta)=\sum\limits_{x^r} logP_{\theta}(x^r)+\sum\limits_{x^u}logP_{\theta}(x^u)
$$

#### Low-density Separation Assumption

接下來介紹一種新的方法，它基於的假設是 Low-density separation

通俗來講，就是這個世界是非黑即白的，在兩個 class 的交界處 data 的密度(density)是很低的，它們之間會有一道明顯的鴻溝，此時 unlabeled data(下圖綠色的點)就是幫助你在原本正確的基礎上挑一條更好的 boundary

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bw.png" width="60%;"/></center>

##### Self Training

low-density separation 最具代表性也最簡單的方法是**self training**

- 先從 labeled data 去訓練一個 model $f^*$，訓練方式沒有限制
- 然後用該$f^*$去對 unlabeled data 打上 label，$y^u=f^*(x^u)$，也叫作 pseudo label
- 從 unlabeled data 中拿出一些 data 加到 labeled data 里，至於 data 的選取需要你自己設計算法來挑選
- 回頭再去訓練$f^*$，循環即可

注：該方法對 Regression 是不適用的

實際上，該方法與之前提到的 generative model 還是挺像的，區別在於：

- Self Training 使用的是 hard label：假設一筆 data 強制屬於某個 class
- Generative Model 使用的是 soft label：假設一筆 data 可以按照概率劃分，不同部分屬於不同 class

如果我們使用的是 neural network 的做法，$\theta^*$是從 labeled data 中得到的一組參數，此時丟進來一個 unlabeled data $x^u$，通過$f^*_{\theta^*}()$後得到$\left [\begin{matrix} 0.7\\ 0.3 \end{matrix}\right ]$，即它有 0.7 的概率屬於 class 1，0.3 的概率屬於 class 2

- 如果此時使用 hard label，則$x^u$的 label 被轉化成$\left [\begin{matrix}1\\ 0 \end{matrix}\right ]$
- 如果此時使用 soft label，則$x^u$的 label 依舊是$\left [\begin{matrix} 0.7\\ 0.3 \end{matrix}\right ]$

可以看到，在 neural network 里使用 soft label 是沒有用的，因為把原始的 model 里的某個點丟回去重新訓練，得到的依舊是同一組參數，實際上 low density separation 就是通過強制分類來提升分類效果的方法

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/self-training.png" width="60%;"/></center>

##### Entropy-based Regularization

該方法是 low-density separation 的進階版，你可能會覺得 hard label 這種直接強制性打標籤的方式有些太武斷了，而 entropy-based regularization 則做了相應的改進：$y^u=f^*_{\theta^*}(x^u)$，其中$y^u$是一個**概率分布(distribution)**

由於我們不知道 unlabeled data $x^u$的 label 到底是什麼，但如果通過 entropy-based regularization 得到的分布集中在某個 class 上的話，那這個 model 就是好的，而如果分布是比較分散的，那這個 model 就是不好的，如下圖所示：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/entropy.png" width="60%;"/></center>

接下來的問題是，如何用數值的方法來 evaluate distribution 的集中(好壞)與否，要用到的方法叫 entropy，一個 distribution 的 entropy 可以告訴你它的集中程度：

$$
E(y^u)=-\sum\limits_{m=1}^5 y_m^u ln(y_m^u)
$$

對上圖中的第 1、2 種情況，算出的$E(y^u)=0$，而第 3 種情況，算出的$E(y^u)=-ln(\frac{1}{5})=ln(5)$，可見 entropy 越大，distribution 就越分散，entropy 越小，distribution 就越集中

因此我們的目標是在 labeled data 上分類要正確，在 unlabeled data 上，output 的 entropy 要越小越好，此時就要修改 loss function

- 對 labeled data 來說，它的 output 要跟正確的 label 越接近越好，用 cross entropy 表示如下：

  $$
  L=\sum\limits_{x^r} C(y^r,\hat y^r)
  $$

- 對 unlabeled data 來說，要使得該 distribution(也就是 output)的 entropy 越小越好：

  $$
  L=\sum\limits_{x^u} E(y^u)
  $$

- 兩項綜合起來，可以用 weight 來加權，以決定哪個部分更為重要一些
  $$
  L=\sum\limits_{x^r} C(y^r,\hat y^r) + \lambda \sum\limits_{x^u} E(y^u)
  $$
  可以發現該式長得很像 regularization，這也就是 entropy regularization 的名稱由來

##### Semi-supervised SVM

SVM 要做的是，給你兩個 class 的 data，去找一個 boundary：

- 要有最大的 margin，讓這兩個 class 分的越開越好
- 要有最小的分類錯誤

對 unlabeled data 窮舉所有可能的 label，下圖中列舉了三種可能的情況；然後對每一種可能的結果都去算 SVM，再找出可以讓 margin 最大，同時又 minimize error 的那種情況，下圖中是用黑色方框標注的情況

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/semi-svm.png" width="60%;"/></center>

SVM paper：Thorsten Joachims, 」_Transductive_ _Inference for Text Classification using Support Vector Machines」,_ ICML, 1999

當然這麼做會存在一個問題，對於 n 筆 unlabeled data，意味著即使在二元分類里也有$2^n$種可能的情況，數據量大的時候，幾乎難以窮舉完畢，上面給出的 paper 提出了一種 approximate 的方法，基本精神是：一開始你先得到一些 label，然後每次改一筆 unlabeled data 的 label，看看可不可以讓你的 objective function 變大，如果變大就去改變該 label，具體內容詳見 paper

#### Smoothness Assumption

##### concepts

smoothness assumption 的基本精神是：近朱者赤，近墨者黑

粗糙的定義是相似的 x 具有相同的$\hat y$，精確的定義是：

- x 的分布是不平均的

- 如果$x^1$和$x^2$在一個 high density region 上很接近的話，那麼$\hat y^1$和$\hat y^2$就是相同的

  也就是這兩個點可以在樣本點高密度集中分布的區域塊中有一條可連接的路徑，即 connected by a high density path

假設下圖是 data 的分布，$x^1,x^2,x^3$是其中的三筆 data，如果單純地看 x 的相似度，顯然$x^2$和$x^3$更接近一些，但對於 smoothness assumption 來說，$x^1$和$x^2$是處於同一塊區域的，它們之間可以有一條相連的路徑；而$x^2$與$x^3$之間則是「斷開」的，沒有 high density path，因此$x^1$與$x^2$更「像」

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/smooth.png" width="60%;"/></center>

##### digits detection

以手寫數字識別為例，對於最右側的 2 和 3 以及最左側的 2，顯然最右側的 2 和 3 在 pixel 上相似度更高一些；但如果把所有連續變化的 2 都放進來，就會產生一種「不直接相連的相似」，根據 Smoothness Assumption 的理論，由於 2 之間有連續過渡的形態，因此第一個 2 和最後一個 2 是比較像的，而最右側 2 和 3 之間由於沒有過渡的 data，因此它們是比較不像的

人臉的過渡數據也同理

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/smooth2.png" width="60%;"/></center>

##### file classification

Smoothness Assumption 在文件分類上是非常有用的

假設對天文學(astronomy)和旅行(travel)的文章進行分類，它們各自有專屬的詞彙，此時如果 unlabeled data 與 label data 的詞彙是相同或重合(overlap)的，那麼就很容易分類；但在真實的情況下，unlabeled data 和 labeled data 之間可能沒有任何重復的 words，因為世界上的詞彙太多了，sparse 的分布很難會使 overlap 發生

但如果 unlabeled data 足夠多，就會以一種相似傳遞的形式，建立起文檔之間相似的橋梁

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/overlap.png" width="60%;"/></center>

##### cluster and then label

在具體實現上，有一種簡單的方法是 cluster and then label，也就是先把 data 分成幾個 cluster，劃分 class 之後再拿去訓練，但這種方法不一定會得到好的結果，因為它的假設是你可以把同一個 class 的樣本點 cluster 在一起，而這其實是沒那麼容易的

對圖像分類來說，如果單純用 pixel 的相似度來劃分 cluster，得到的結果一般都會很差，你需要設計一個很好的方法來描述 image(類似 Deep Autoencoder 的方式來提取 feature)，這樣 cluster 才會有效果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cluster.png" width="60%;"/></center>

##### Graph-based Approach

之前講的是比較直覺的做法，接下來引入 Graph Structure 來表達 connected by a high density path 這件事

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/graph.png" width="60%;"/></center>

我們把所有的 data points 都建成一個 graph，有時候建立 vertex 之間的關係是比較容易的，比如網頁之間的鏈接關係、論文之間的引用關係；但有時候需要你自己去尋找 vertex 之間的關係

graph 的好壞，對結果起著至關重要的影響，而如何 build graph 卻是一件 heuristic 的事情，需要憑著經驗和直覺來做

- 首先定義兩個 object $x^i,x^j$之間的相似度 $s(x^i, x^j)$

  如果是基於 pixel 的相似度，performance 可能會不太好；建議使用 autoencoder 提取出來的 feature 來計算相似度，得到的 performance 會好一些

- 算完相似度後，就可以建 graph 了，方式有很多種：

  - k nearest neighbor：假設 k=3，則每個 point 與相似度最接近的 3 個點相連
  - e-neighborhood：每個 point 與相似度超過某個特定 threshold e 的點相連

- 除此之外，還可以給 Edge 特定的 weight，讓它與相似度$s(x^i,x^j)$成正比

  - 建議用 RBM function 來確定相似度：$s(x^i,x^j)=e^{-\gamma||x^i-x^j||^2 }$

    這裡$x^i,x^j$均為 vector，計算它們的 Euclidean Distance(歐幾里得距離)，加上參數後再去 exponential

  - 至於加 exponential，經驗上來說通常是可以幫助提升 performance 的，在這裡只有當$x^i,x^j$非常接近的時候，singularity 才會大；只要距離稍微遠一點，singularity 就會下降得很快，變得很小
  - 使用 exponential 的 RBM function 可以做到只有非常近的兩個點才能相連，稍微遠一點就無法相連的效果，避免了下圖中跨區域相連的情況

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/build-graph.png" width="60%;"/></center>

graph-based approach 的基本精神是，在 graph 上已經有一些 labeled data，那麼跟它們相連的 point，屬於同一類的概率就會上升，每一筆 data 都會去影響它的鄰居，而 graph 帶來的最重要的好處是，這個影響是會隨著 edges**傳遞**出去的，即使有些點並沒有真的跟 labeled data 相連，也可以被傳遞到相應的屬性

比如下圖中，如果 graph 建的足夠好，那麼兩個被分別 label 為藍色和紅色的點就可以傳遞完兩張完整的圖；從中我們也可以看出，如果想要讓這種方法生效，收集到的 data 一定要足夠多，否則可能傳遞到一半，graph 就斷掉了，information 的傳遞就失效了

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/graph-nei.png" width="60%;"/></center>

介紹完了如何定性使用 graph，接下來介紹一下如何定量使用 graph

定量的使用方式是定義 label 的 smoothness，下圖中，edge 上的數字是 weight，$x^i$表達 data，$y^i$表示 data 的 label，計算 smoothness 的方式為：

$$
S=\frac{1}{2}\sum\limits_{i,j} w_{i,j}(y^i-y^j)^2
$$

**我們期望 smooth 的值越小越好**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/graph-cal.png" width="60%;"/></center>

當然上面的式子還可以化簡，如果把 labeled data 和 unlabeled data 的 y 組成一個(R+U)-dim vector，即

$$
y=\left [\begin{matrix}
...y^i...y^j
\end{matrix} \right ]^T
$$

於是 smooth 可以改寫為：

$$
S=\frac{1}{2}\sum\limits_{i,j} w_{i,j}(y^i-y^j)^2=y^TLy
$$

其中 L 為(R+U)×(R+U) matrix，成為**Graph Laplacian**， 定義為$L=D-W$

- W：把 data point 兩兩之間 weight 的關係建成 matrix，代表了$x^i$與$x^j$之間的 weight 值
- D：把 W 的每一個 row 上的值加起來放在該行對應的 diagonal 上即可，比如 5=2+3,3=2+1,...

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/graph-cal2.png" width="60%;"/></center>

對$S=y^TLy$來說，y 是 label，是 neural network 的 output，取決於 neural network 的 parameters，因此要在原來僅針對 labeled data 的 loss function 中加上這一項，得到：

$$
L=\sum\limits_{x^r}C(y^r,\hat y^r) + \lambda S
$$

$\lambda S$實際上也是一個 regularization term

訓練目標：

- labeled data 的 cross entropy 越小越好(neural network 的 output 跟真正的 label 越接近越好)
- smooth S 越小越好(neural network 的 output，不管是 labeled 還是 unlabeled，都要符合 Smoothness Assumption 的假設)

具體訓練的時候，不一定只局限於 neural network 的 output 要 smooth，可以對中間任意一個 hidden layer 加上 smooth 的限制

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/graph-cal3.png" width="60%;"/></center>

#### Better Representation

Better Representation 的精神是，去蕪存菁，化繁為簡

我們觀察到的世界是比較複雜的，而在它的背後其實是有一些比較簡單的東西，在操控著這個複雜的世界，所以只要你能夠看透這個世界的假象，直指它的核心的話，就可以讓 training 變得比較容易

舉一個例子，在神雕俠侶中，楊過要在三招之內剪掉樊一翁的鬍子，雖然鬍子的變化是比較複雜的，但頭的變化是有限的，楊過看透了這一件事情就可以把鬍子剪掉。在這個例子中，樊一翁的鬍子就是 original representation，而他的頭就是你要找的 better representation

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/better-re.png" width="60%;"/></center>

算法具體思路和內容到 unsupervised learning 的時候再介紹
